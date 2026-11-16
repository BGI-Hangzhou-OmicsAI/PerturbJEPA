"""Parse10M sparse preprocessing; all paths are explicit and source data is read-only."""

import argparse
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import resource
import shutil
import time

import h5py
import numpy as np
from scipy import sparse


DATA_ROOT = Path(__file__).resolve().parents[3] / "data/parse10m"
CELL_TYPES = ['CD14 Mono', 'CD4 Memory', 'CD4 Naive', 'CD8 Memory', 'CD8 Naive', 'NK', 'B Naive']
FORMAT = 'parse10m-csr-memmap-v1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_fingerprint(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def safe_names(labels):
    names = [re.sub(r'[^A-Za-z0-9._-]', '_', label) for label in labels]
    if len(set(names)) != len(names) or any(name in ('', '.', '..') for name in names):
        raise ValueError('Labels produce ambiguous or unsafe directory names.')
    return names


class Source:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with h5py.File(self.path, 'r') as data:
            if data['X'].attrs.get('encoding-type') != 'csr_matrix':
                raise ValueError('X must be CSR.')
            self.shape = tuple(int(value) for value in data['X'].attrs['shape'])
            self.genes = data['var/_index'].asstr()[:].astype(str)
            self.indptr = data['X/indptr'][:].astype(np.int64)
            if len(self.genes) != self.shape[1] or len(self.indptr) != self.shape[0] + 1:
                raise ValueError('Invalid source dimensions.')
            if self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0):
                raise ValueError('Invalid CSR row pointers.')
            if self.indptr[-1] != len(data['X/data']) or len(data['X/data']) != len(data['X/indices']):
                raise ValueError('Invalid CSR array lengths.')
            self.labels, self.codes = {}, {}
            for key in ('donor', 'cytokine', 'cell_type', 'bc1_well'):
                self.labels[key] = data[f'obs/{key}/categories'].asstr()[:].tolist()
                codes = data[f'obs/{key}/codes'][:].astype(np.int32)
                if len(codes) != self.shape[0] or np.any(codes < 0) or np.any(codes >= len(self.labels[key])):
                    raise ValueError(f'Missing or invalid {key} labels.')
                if len(set(self.labels[key])) != len(self.labels[key]):
                    raise ValueError(f'Duplicate {key} categories.')
                self.codes[key] = codes
        self.group_shape = tuple(len(self.labels[key]) for key in ('donor', 'cytokine', 'cell_type'))
        self.group_ids = np.ravel_multi_index(
            tuple(self.codes[key] for key in ('donor', 'cytokine', 'cell_type')), self.group_shape
        )

    def chunks(self, chunk_rows, chunk_nnz, start=0):
        while start < self.shape[0]:
            end = min(start + chunk_rows, self.shape[0])
            limit = int(self.indptr[start]) + chunk_nnz
            end = min(end, max(start + 1, int(np.searchsorted(self.indptr, limit, side='right')) - 1))
            yield start, end
            start = end

    def read_chunk(self, data, start, end, genes=None):
        begin, finish = int(self.indptr[start]), int(self.indptr[end])
        raw_values = data['X/data'][begin:finish]
        indices = data['X/indices'][begin:finish]
        if np.issubdtype(raw_values.dtype, np.integer):
            if np.issubdtype(raw_values.dtype, np.signedinteger) and np.any(raw_values < 0):
                raise ValueError('Source X contains negative counts.')
        elif np.any(~np.isfinite(raw_values)) or np.any(raw_values < 0) or np.any(raw_values != np.floor(raw_values)):
            raise ValueError('Source X must contain finite, nonnegative integer counts.')
        values = raw_values.astype(np.float64)
        if np.any(indices < 0) or np.any(indices >= self.shape[1]):
            raise ValueError('Invalid source gene indices.')
        matrix = sparse.csr_matrix((values, indices, self.indptr[start:end + 1] - begin),
                                   shape=(end - start, self.shape[1]))
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if genes is not None:
            matrix = matrix[:, genes]
        matrix.sort_indices()
        return matrix


def global_detection(source, cache_path, chunk_rows, chunk_nnz):
    if cache_path is not None:
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache['metadata']))
            expected = source_fingerprint(source.path)
            fingerprint = {'path': str(Path(metadata['source']).resolve()),
                           'bytes': metadata['sourceBytes'], 'mtime_ns': metadata['sourceMtimeNs']}
            if fingerprint != expected or metadata['shape'] != list(source.shape):
                raise ValueError('Gene detection cache is stale; use a valid cache or --scan-genes.')
            if metadata['genes'] != source.genes.tolist():
                raise ValueError('Gene cache ordering mismatch.')
            if metadata['donors'] != source.labels['donor'] or metadata['types'] != source.labels['cell_type']:
                raise ValueError('Gene cache categories mismatch.')
            detected = cache['detected']
            group_counts = cache['group_counts']
            expected_counts = np.bincount(
                source.codes['donor'] * len(source.labels['cell_type']) + source.codes['cell_type'],
                minlength=len(source.labels['donor']) * len(source.labels['cell_type'])
            )
            if detected.shape != (len(expected_counts), source.shape[1]) or not np.array_equal(group_counts, expected_counts):
                raise ValueError('Gene cache does not cover all source cells.')
            if np.any(detected > expected_counts[:, None]):
                raise ValueError('Invalid gene cache detection counts.')
            print('Using validated full-source gene detection cache.', flush=True)
            return detected.sum(axis=0, dtype=np.uint64)
    detected = np.zeros(source.shape[1], dtype=np.uint64)
    last_report = time.monotonic()
    with h5py.File(source.path, 'r') as data:
        for start, end in source.chunks(chunk_rows, chunk_nnz):
            matrix = source.read_chunk(data, start, end)
            detected += np.bincount(matrix.indices, minlength=source.shape[1]).astype(np.uint64)
            if time.monotonic() - last_report > 20:
                print(f'Gene detection: {end:,}/{source.shape[0]:,} cells', flush=True)
                last_report = time.monotonic()
    return detected


def initialize_scan(source_path, gene_columns):
    global scan_source, scan_file, scan_genes
    scan_source = Source(source_path)
    scan_file = h5py.File(source_path, 'r')
    scan_genes = gene_columns


def scan_row_counts(bounds):
    start, end = bounds
    matrix = scan_source.read_chunk(scan_file, start, end, scan_genes)
    return start, end, np.diff(matrix.indptr).astype(np.int32)


def initialize_export(source_path, genes, keep, row_nnz_path, target_sum):
    global scan_keep, scan_row_nnz, scan_target
    initialize_scan(source_path, genes)
    scan_keep = keep
    scan_row_nnz = np.load(row_nnz_path, mmap_mode='r', allow_pickle=False)
    scan_target = target_sum


def scan_export(bounds):
    start, end = bounds
    chosen = np.flatnonzero(scan_keep[scan_source.group_ids[start:end]] & (scan_row_nnz[start:end] > 0))
    if not len(chosen):
        return start, end, chosen, None
    matrix = scan_source.read_chunk(scan_file, start, end, scan_genes)[chosen]
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(sums <= 0):
        raise ValueError('Unexpected zero-library cell during export.')
    matrix.data *= np.repeat(scan_target / sums, np.diff(matrix.indptr))
    np.log1p(matrix.data, out=matrix.data)
    matrix.data = matrix.data.astype(np.float32)
    return start, end, chosen, matrix


def export_chunks(args, source, genes, keep, next_row):
    bounds = iter(source.chunks(args.chunk_rows, args.chunk_nnz, next_row))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_export,
                             initargs=(source.path, genes, keep, args.output / '_plan/row_nnz.npy', args.target_sum)) as pool:
        pending = deque()
        for _ in range(args.workers):
            chunk = next(bounds, None)
            if chunk is not None:
                pending.append(pool.submit(scan_export, chunk))
        while pending:
            yield pending.popleft().result()
            chunk = next(bounds, None)
            if chunk is not None:
                pending.append(pool.submit(scan_export, chunk))


def plan(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / '_plan'
    work.mkdir(exist_ok=True)
    config = {'format': FORMAT, 'source': source_fingerprint(args.source.resolve()),
              'cell_types': args.cell_types, 'min_detected': args.min_detected,
              'min_cells': args.min_cells, 'target_sum': args.target_sum, 'control': args.control}
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (work / 'config.json').exists() and read_json(work / 'config.json') != config:
        raise ValueError('Output belongs to a different configuration/source; choose another output directory.')
    if not (work / 'config.json').exists() and (output / 'manifest.json').exists():
        raise ValueError('Existing manifest without matching plan; refusing to overwrite.')
    write_json(work / 'config.json', config)
    if (work / '_SUCCESS').exists():
        manifest = read_json(output / 'manifest.json')
        if manifest['config_id'] != config_id:
            raise ValueError('Plan fingerprint mismatch.')
        print('Reusing completed plan.', flush=True)
        return manifest
    source = Source(args.source)
    if len(set(args.cell_types)) != len(args.cell_types) or not set(args.cell_types) <= set(source.labels['cell_type']):
        raise ValueError('Selected cell types must be unique and present in source.')
    control_index = source.labels['cytokine'].index(args.control)
    detected = global_detection(source, args.gene_cache, args.chunk_rows, args.chunk_nnz)
    genes = np.flatnonzero(detected >= args.min_detected)
    if not len(genes):
        raise ValueError('Gene filter retained no genes.')
    np.save(output / 'genes.npy', source.genes[genes])
    np.save(output / 'gene_source_columns.npy', genes)
    np.save(work / 'global_detected.npy', detected)
    row_nnz = np.lib.format.open_memmap(work / 'row_nnz.npy', mode='w+', dtype=np.int32, shape=(source.shape[0],))
    last_report = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_scan, initargs=(source.path, genes)) as pool:
        for start, end, counts_chunk in pool.map(scan_row_counts, source.chunks(args.chunk_rows, args.chunk_nnz), chunksize=4):
            row_nnz[start:end] = counts_chunk
            if time.monotonic() - last_report > 20:
                print(f'Plan: {end:,}/{source.shape[0]:,} cells', flush=True)
                last_report = time.monotonic()
    row_nnz.flush()
    valid = row_nnz > 0
    counts = np.bincount(source.group_ids[valid], minlength=int(np.prod(source.group_shape))).reshape(source.group_shape)
    np.save(work / 'group_counts.npy', counts)
    nnz = np.zeros(int(np.prod(source.group_shape)), dtype=np.int64)
    np.add.at(nnz, source.group_ids, row_nnz)
    nnz = nnz.reshape(source.group_shape)
    donors, conditions, types = (source.labels[key] for key in ('donor', 'cytokine', 'cell_type'))
    selected = np.isin(types, args.cell_types)
    paired = (counts >= args.min_cells) & (counts[:, control_index, None, :] >= args.min_cells)
    paired &= selected[None, None, :]
    paired[:, control_index, :] = False
    keep = paired.copy()
    keep[:, control_index, :] = paired.any(axis=1)
    donor_names, condition_names, type_names = (safe_names(labels) for labels in (donors, conditions, types))
    blocks, pairs, rejected = [], [], []
    for donor_index, donor in enumerate(donors):
        for condition_index, condition in enumerate(conditions):
            for type_index in np.flatnonzero(selected):
                coordinate = (donor_index, condition_index, int(type_index))
                group_id = int(np.ravel_multi_index(coordinate, source.group_shape))
                n_cells = int(counts[coordinate])
                n_control = int(counts[donor_index, control_index, type_index])
                if keep[coordinate]:
                    blocks.append({'id': group_id, 'donor': donor, 'condition': condition,
                                   'cell_type': types[type_index], 'n_cells': n_cells,
                                   'nnz': int(nnz[coordinate]), 'shape': [n_cells, len(genes)],
                                   'path': '/'.join((donor_names[donor_index], condition_names[condition_index], type_names[type_index]))})
                if condition_index == control_index:
                    continue
                record = {'donor': donor, 'condition': condition, 'cell_type': types[type_index],
                          'n_cells': n_cells, 'n_control': n_control}
                if paired[coordinate]:
                    record.update({'treated_id': group_id, 'control_id': int(np.ravel_multi_index(
                        (donor_index, control_index, int(type_index)), source.group_shape))})
                    pairs.append(record)
                else:
                    record['reasons'] = [reason for reason, insufficient in (
                        ('insufficient_control', n_control < args.min_cells),
                        ('insufficient_treated', n_cells < args.min_cells)) if insufficient]
                    rejected.append(record)
    well_shape = (len(donors), len(source.labels['bc1_well']), len(types))
    well_ids = np.ravel_multi_index(tuple(source.codes[key] for key in ('donor', 'bc1_well', 'cell_type')), well_shape)
    raw_counts = np.bincount(well_ids, minlength=int(np.prod(well_shape))).reshape(well_shape)
    retained = valid & keep.ravel()[source.group_ids]
    retained_counts = np.bincount(well_ids[retained], minlength=int(np.prod(well_shape))).reshape(well_shape)
    np.save(work / 'well_counts_raw.npy', raw_counts)
    np.save(work / 'well_counts_retained.npy', retained_counts)
    well_condition = np.full(well_shape[:2], -1, dtype=np.int32)
    mapping_ids = (source.codes['donor'].astype(np.int64) * well_shape[1] + source.codes['bc1_well']) * len(conditions) + source.codes['cytokine']
    for mapping_id in np.unique(mapping_ids):
        donor_well, condition_index = divmod(int(mapping_id), len(conditions))
        donor_index, well_index = divmod(donor_well, well_shape[1])
        if well_condition[donor_index, well_index] != -1:
            raise ValueError('A donor/well maps to multiple conditions; an explicit plate identifier is required.')
        well_condition[donor_index, well_index] = condition_index
    np.save(work / 'well_condition.npy', well_condition)
    retained_nnz = sum(block['nnz'] for block in blocks)
    retained_cells = sum(block['n_cells'] for block in blocks)
    manifest = {'format': FORMAT, 'config_id': config_id, 'config': config,
                'created': datetime.now(timezone.utc).isoformat(), 'categories': source.labels,
                'directory_names': {'donor': donor_names, 'condition': condition_names, 'cell_type': type_names},
                'stats': {'source_cells': source.shape[0], 'source_genes': source.shape[1],
                          'retained_genes': len(genes), 'zero_library_cells': int((~valid).sum()),
                          'selected_type_cells_before_pair_filter': int(np.count_nonzero(valid & selected[source.codes['cell_type']])),
                          'retained_cells': retained_cells, 'retained_nnz': retained_nnz,
                          'eligible_pairs': len(pairs), 'rejected_pairs': len(rejected),
                          'control_pools': sum(block['condition'] == args.control for block in blocks),
                          'estimated_expression_bytes': retained_nnz * 8 + retained_cells * 20 + len(blocks) * 8},
                'blocks': blocks, 'pairs': pairs, 'rejected': rejected}
    if source_fingerprint(source.path) != config['source']:
        raise ValueError('Source changed during planning.')
    write_json(output / 'manifest.json', manifest)
    (work / '_SUCCESS').write_text(config_id + '\n')
    print(json.dumps(manifest['stats'], indent=2), flush=True)
    return manifest


ARRAYS = {'values': np.float32, 'indices': np.int32, 'indptr': np.int64,
          'source_rows': np.int64, 'well_codes': np.int32}


class Writers:
    def __init__(self, output, blocks):
        self.output, self.blocks = output, blocks
        soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        self.limit = min(len(blocks), max(1, (soft_limit - 256) // len(ARRAYS)))
        self.cache = OrderedDict()
        self.buffers = {}
        self.pool = ThreadPoolExecutor(max_workers=8 if self.limit == len(blocks) else 1)

    def get(self, group_id):
        if group_id in self.cache:
            self.cache.move_to_end(group_id)
            return self.cache[group_id]
        if len(self.cache) >= self.limit:
            _, arrays = self.cache.popitem(last=False)
            for handle, offset in arrays.values():
                handle.close()
        directory = self.output / self.blocks[group_id]['path']
        arrays = {}
        for name in ARRAYS:
            handle = (directory / f'{name}.npy').open('r+b', buffering=0)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                np.lib.format.read_array_header_2_0(handle)
            else:
                handle.close()
                raise ValueError('Unsupported output NPY version.')
            arrays[name] = (handle, handle.tell())
        self.cache[group_id] = arrays
        return arrays

    def append(self, group_id, matrix, source_rows, well_codes, row_start, value_start):
        if group_id not in self.buffers:
            self.buffers[group_id] = {'row_start': row_start, 'value_start': value_start,
                                      'parts': {name: [] for name in ARRAYS}}
        parts = self.buffers[group_id]['parts']
        parts['values'].append(matrix.data)
        parts['indices'].append(matrix.indices)
        parts['indptr'].append(matrix.indptr[1:].astype(np.int64) + value_start)
        parts['source_rows'].append(source_rows)
        parts['well_codes'].append(well_codes)

    def write_buffer(self, item):
        group_id, buffer = item
        arrays = self.get(group_id)
        for name, pieces in buffer['parts'].items():
            values = np.concatenate(pieces).astype(ARRAYS[name], copy=False)
            start = buffer['value_start'] if name in ('values', 'indices') else buffer['row_start'] + (name == 'indptr')
            handle, offset = arrays[name]
            payload = memoryview(values).cast('B')
            position = offset + start * values.dtype.itemsize
            while len(payload):
                written = os.pwrite(handle.fileno(), payload, position)
                if written <= 0:
                    raise OSError('Incomplete sparse block write.')
                payload = payload[written:]
                position += written
        buffer['parts'].clear()

    def flush(self):
        for result in self.pool.map(self.write_buffer, list(self.buffers.items())):
            pass
        self.buffers.clear()

    def close(self):
        self.flush()
        self.pool.shutdown()
        for arrays in self.cache.values():
            for handle, offset in arrays.values():
                handle.close()
        self.cache.clear()


def export(args, manifest):
    output = args.output.resolve()
    if (output / 'export.SUCCESS').exists():
        if (output / 'export.SUCCESS').read_text().strip() != manifest['config_id']:
            raise ValueError('Export fingerprint mismatch.')
        print('Reusing completed sparse export.', flush=True)
        return
    source = Source(args.source)
    if source_fingerprint(source.path) != manifest['config']['source']:
        raise ValueError('Source changed since planning.')
    blocks = {block['id']: block for block in manifest['blocks']}
    if not blocks:
        raise ValueError('No eligible combinations; inspect manifest and thresholds.')
    progress_path = output / '_plan/export_progress.json'
    group_total = int(np.prod(source.group_shape))
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress['config_id'] != manifest['config_id']:
            raise ValueError('Export checkpoint mismatch.')
        next_row = progress['next_row']
        row_cursor = np.array(progress['row_cursor'], dtype=np.int64)
        nnz_cursor = np.array(progress['nnz_cursor'], dtype=np.int64)
    else:
        required = manifest['stats']['estimated_expression_bytes']
        if shutil.disk_usage(output).free < required * 1.05:
            raise OSError(f'Insufficient disk space: export needs approximately {required:,} bytes.')
        for donor in manifest['directory_names']['donor']:
            for condition in manifest['directory_names']['condition']:
                (output / donor / condition).mkdir(parents=True, exist_ok=True)
        for block_number, block in enumerate(blocks.values(), start=1):
            directory = output / block['path']
            directory.mkdir(parents=True, exist_ok=True)
            for name, dtype in ARRAYS.items():
                length = block['nnz'] if name in ('values', 'indices') else block['n_cells'] + (name == 'indptr')
                array = np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+', dtype=dtype, shape=(length,))
                if name == 'indptr':
                    array[0] = 0
                array.flush()
                array._mmap.close()
            write_json(directory / 'metadata.json', dict(block, format=FORMAT,
                       expression_transform='normalize_total_log1p', target_sum=args.target_sum,
                       well_categories=source.labels['bc1_well'], config_id=manifest['config_id']))
            if block_number % 100 == 0:
                print(f'Preparing sparse blocks: {block_number:,}/{len(blocks):,}', flush=True)
        next_row = 0
        row_cursor = np.zeros(group_total, dtype=np.int64)
        nnz_cursor = np.zeros(group_total, dtype=np.int64)
        write_json(progress_path, {'config_id': manifest['config_id'], 'next_row': 0,
                                  'row_cursor': row_cursor.tolist(), 'nnz_cursor': nnz_cursor.tolist()})
    genes = np.load(output / 'gene_source_columns.npy', allow_pickle=False)
    keep = np.zeros(group_total, dtype=bool)
    keep[list(blocks)] = True
    writers = Writers(output, blocks)
    last_report = time.monotonic()
    chunks_since_checkpoint = 0
    try:
        for start, end, chosen, matrix in export_chunks(args, source, genes, keep, next_row):
                group_ids = source.group_ids[start:end]
                if len(chosen):
                    for group_id in np.unique(group_ids[chosen]):
                        positions = np.flatnonzero(group_ids[chosen] == group_id)
                        piece = matrix[positions]
                        source_rows = chosen[positions] + start
                        row_start, value_start = int(row_cursor[group_id]), int(nnz_cursor[group_id])
                        row_end, value_end = row_start + len(positions), value_start + piece.nnz
                        if row_end > blocks[group_id]['n_cells'] or value_end > blocks[group_id]['nnz']:
                            raise ValueError('Export exceeds planned block sizes.')
                        writers.append(int(group_id), piece, source_rows, source.codes['bc1_well'][source_rows],
                                       row_start, value_start)
                        row_cursor[group_id], nnz_cursor[group_id] = row_end, value_end
                chunks_since_checkpoint += 1
                if chunks_since_checkpoint >= args.checkpoint_chunks or end == source.shape[0]:
                    writers.flush()
                    write_json(progress_path, {'config_id': manifest['config_id'], 'next_row': end,
                                              'row_cursor': row_cursor.tolist(), 'nnz_cursor': nnz_cursor.tolist()})
                    chunks_since_checkpoint = 0
                if time.monotonic() - last_report > 20:
                    print(f'Export: {end:,}/{source.shape[0]:,} source cells', flush=True)
                    last_report = time.monotonic()
    finally:
        writers.close()
    for group_id, block in blocks.items():
        if row_cursor[group_id] != block['n_cells'] or nnz_cursor[group_id] != block['nnz']:
            raise ValueError(f'Incomplete block: {block["path"]}')
    if source_fingerprint(source.path) != manifest['config']['source']:
        raise ValueError('Source changed during export.')
    (output / 'export.SUCCESS').write_text(manifest['config_id'] + '\n')
    print('Sparse mmap export complete.', flush=True)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('stage', choices=['plan', 'export', 'run'])
    command.add_argument('--source', type=Path, default=DATA_ROOT / 'Parse10M.h5ad')
    command.add_argument('--output', type=Path, default=DATA_ROOT / 'v1')
    command.add_argument('--gene-cache', type=Path, default=None)
    command.add_argument('--scan-genes', action='store_true', help='Compute exact detection from X instead of using cache.')
    command.add_argument('--cell-types', nargs='+', default=CELL_TYPES)
    command.add_argument('--control', default='PBS')
    command.add_argument('--min-detected', type=int, default=1000)
    command.add_argument('--min-cells', type=int, default=256)
    command.add_argument('--target-sum', type=float, default=10000)
    command.add_argument('--chunk-rows', type=int, default=2048)
    command.add_argument('--chunk-nnz', type=int, default=4000000)
    command.add_argument('--checkpoint-chunks', type=int, default=32)
    command.add_argument('--workers', type=int, default=2, help='Parallel source scanning processes.')
    return command


def main():
    args = parser().parse_args()
    if min(args.min_detected, args.min_cells, args.chunk_rows, args.chunk_nnz, args.checkpoint_chunks, args.workers) < 1 or not np.isfinite(args.target_sum) or args.target_sum <= 0:
        raise ValueError('Thresholds, target sum and chunk sizes must be positive.')
    if args.scan_genes:
        args.gene_cache = None
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.preprocess.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process is writing to this output directory.') from error
        manifest = plan(args)
        if args.stage in ('export', 'run'):
            export(args, manifest)


if __name__ == '__main__':
    main()
