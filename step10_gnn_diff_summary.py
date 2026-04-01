import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, '/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code')
import step10_hole as s10
from step10_gnn import gnn_merge_and_assign
from step10_gnn_assign import _gnn_greedy_assign


def load_raw_instances_and_masks(json_path, data, image_h, image_w):
    raw_instances = []
    raw_instance_masks = []
    union_mask = np.zeros((image_h, image_w), dtype=np.uint8)

    stem_base = json_path.stem.replace('_instances_with_polygon', '')
    step7_mask_path = s10.STEP7_MASK_DIR / f'{stem_base}_instance_mask.png'

    if step7_mask_path.exists():
        inst_mask_img = np.array(Image.open(str(step7_mask_path)))
        if inst_mask_img.ndim == 3:
            inst_mask_img = inst_mask_img[:, :, 0]
        if inst_mask_img.shape[0] != image_h or inst_mask_img.shape[1] != image_w:
            inst_mask_img = s10.cv2.resize(inst_mask_img.astype(np.uint16), (image_w, image_h), interpolation=s10.cv2.INTER_NEAREST)

        unique_ids = np.unique(inst_mask_img)
        unique_ids = unique_ids[unique_ids > 0]
        raw_idx = 0
        for iid in unique_ids:
            binary = (inst_mask_img == iid).astype(np.uint8)
            props = s10.mask_to_props(binary)
            if props is None:
                continue
            raw_instance_masks.append(binary)
            union_mask = np.maximum(union_mask, binary)
            raw_instances.append({
                'instance_index': raw_idx,
                'label': 'lettuce',
                'shape_type': 'mask',
                'centroid': props['centroid'],
                'bbox': props['bbox'],
                'area': props['area'],
            })
            raw_idx += 1
    else:
        if isinstance(data.get('instances', None), list) and len(data['instances']) > 0:
            raw_idx = 0
            for ins in data['instances']:
                poly = ins.get('polygon', None)
                mask = s10.polygons_to_mask(poly, image_h, image_w)
                props = s10.mask_to_props(mask)
                if props is None:
                    continue
                raw_instance_masks.append(mask.astype(np.uint8))
                union_mask = np.maximum(union_mask, mask.astype(np.uint8))
                raw_instances.append({
                    'instance_index': raw_idx,
                    'label': str(ins.get('category_name', '')),
                    'shape_type': 'polygon',
                    'centroid': props['centroid'],
                    'bbox': props['bbox'],
                    'area': props['area'],
                })
                raw_idx += 1
        else:
            shapes = data.get('shapes', [])
            raw_idx = 0
            for shape in shapes:
                st = shape.get('shape_type', 'polygon')
                if st not in ['polygon', 'rectangle']:
                    continue
                mask = s10.shape_to_mask(shape, image_h, image_w)
                props = s10.mask_to_props(mask)
                if props is None:
                    continue
                raw_instance_masks.append(mask.astype(np.uint8))
                union_mask = np.maximum(union_mask, mask.astype(np.uint8))
                raw_instances.append({
                    'instance_index': raw_idx,
                    'label': shape.get('label', ''),
                    'shape_type': st,
                    'centroid': props['centroid'],
                    'bbox': props['bbox'],
                    'area': props['area'],
                })
                raw_idx += 1

    return raw_instances, raw_instance_masks, union_mask


def apply_manual_empty_constraint(stem_base, hole_centers, matches, avg_spacing):
    ann_json = s10.find_ann_json_for_stem(stem_base, s10.ANN_DIR)
    manual_empty_hole_ids = set()
    if ann_json is not None:
        manual_pts = s10.load_manual_empty_hole_points(ann_json)
        manual_empty_hole_ids = s10.map_manual_empty_points_to_holes(manual_pts, hole_centers, avg_spacing=avg_spacing)

    if manual_empty_hole_ids:
        matches = [m for m in matches if int(m['hole_index']) not in manual_empty_hole_ids]

    matched_holes = {int(m['hole_index']) for m in matches}
    empty_holes = []
    for hh in hole_centers:
        hid = int(hh['hole_index'])
        if (hid not in matched_holes) or (hid in manual_empty_hole_ids):
            empty_holes.append(hid)

    return matches, empty_holes


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    json_files = sorted(s10.INPUT_DIR.glob('*.json'))

    out_dir = Path('/ldap_shared/home/t_jyy/ssd/home/jiangyiyu/unit3code/mydata/output/step10_graph_merge_area_rank')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'gnn_before_after_summary.csv'

    rows = []

    for jp in json_files:
        stem = jp.stem
        stem_base = stem.replace('_instances_with_polygon', '')
        try:
            data = s10.load_json(jp)
            img_path = s10.find_image_for_json(jp, data)
            if img_path is None:
                rows.append({
                    'image_stem': stem,
                    'status': 'failed_no_image',
                    'before_matches': '', 'before_empty_holes': '',
                    'after_matches': '', 'after_empty_holes': '',
                    'delta_matches': '', 'delta_empty_holes': ''
                })
                continue

            image = s10.read_image(img_path)
            h, w = image.shape[:2]

            raw_instances, raw_masks, union_mask = load_raw_instances_and_masks(jp, data, h, w)
            if len(raw_instances) < 5:
                rows.append({
                    'image_stem': stem,
                    'status': 'failed_too_few_instances',
                    'before_matches': '', 'before_empty_holes': '',
                    'after_matches': '', 'after_empty_holes': '',
                    'delta_matches': '', 'delta_empty_holes': ''
                })
                continue

            raw_centroids_xy = np.array([ins['centroid'] for ins in raw_instances], dtype=np.float64)
            origin, e_u, e_v, _ = s10.estimate_tray_axes(raw_centroids_xy, union_mask)
            local_uv = s10.world_to_local(raw_centroids_xy, origin, e_u, e_v)
            u_vals = local_uv[:, 0]
            v_vals = local_uv[:, 1]
            col_centers_u, _, _ = s10.fit_equal_spacing_1d_edge_aware(u_vals, s10.NUM_COLS, device=device)
            row_centers_v, _, _ = s10.fit_equal_spacing_1d_edge_aware(v_vals, s10.NUM_ROWS, device=device)
            hole_centers = s10.build_theory_centers(row_centers_v, col_centers_u, origin, e_u, e_v)
            avg_col_spacing, avg_row_spacing, avg_spacing = s10.compute_average_spacing(hole_centers, s10.NUM_ROWS, s10.NUM_COLS)
            match_radius = s10.choose_match_radius(avg_col_spacing, avg_row_spacing)

            adj, dmat = s10.compute_adjacency(raw_centroids_xy, s10.GRAPH_MAX_NEIGHBOR_DIST, s10.GRAPH_KNN, device=device)
            hole_assign_rows = s10.assign_instances_to_candidate_holes(raw_instances, hole_centers, match_radius, s10.NUM_ROWS, s10.NUM_COLS, device=device)

            # ---- Before (rule merge) ----
            groups_b, _ = s10.graph_rule_merge(raw_instances, raw_masks, hole_assign_rows, adj, dmat)
            opt_b = []
            for _, gidx in groups_b.items():
                m = s10.merge_instance_group(gidx, raw_instances, raw_masks)
                if m is not None:
                    opt_b.append(m)
            for i, ins in enumerate(opt_b):
                ins['opt_instance_index'] = i

            matches_b, _ = s10.greedy_match_optimized_instances_with_mask_overlap(
                opt_instances=opt_b,
                hole_centers=hole_centers,
                base_radius=match_radius,
                num_rows=s10.NUM_ROWS,
                num_cols=s10.NUM_COLS,
                image_h=h,
                image_w=w,
                device=device,
            )
            matches_b, empty_b = apply_manual_empty_constraint(stem_base, hole_centers, matches_b, avg_spacing)

            # ---- After (GNN merge) ----
            try:
                groups_g, hole_logits = gnn_merge_and_assign(
                    instances=raw_instances,
                    masks=raw_masks,
                    har=hole_assign_rows,
                    adj=adj,
                    dmat=dmat,
                    holes=hole_centers,
                    ih=h,
                    iw=w,
                    device=device,
                    n_iter=20,
                )

                opt_g = []
                group_raw_indices = []
                for _, gidx in groups_g.items():
                    m = s10.merge_instance_group(gidx, raw_instances, raw_masks)
                    if m is not None:
                        opt_g.append(m)
                        group_raw_indices.append(gidx)
                for i, ins in enumerate(opt_g):
                    ins['opt_instance_index'] = i

                # aggregate node->hole logits to opt-instance->hole logits
                if len(opt_g) > 0 and hole_logits is not None and hole_logits.size > 0:
                    opt_hole_scores = np.zeros((len(opt_g), len(hole_centers)), dtype=np.float32)
                    for oi, gidx in enumerate(group_raw_indices):
                        idx = np.array(gidx, dtype=np.int32)
                        if idx.size == 0:
                            continue
                        sub = hole_logits[idx]
                        if sub.size == 0:
                            continue
                        if sub.ndim == 1:
                            opt_hole_scores[oi] = sub
                        else:
                            opt_hole_scores[oi] = np.max(sub, axis=0)
                else:
                    opt_hole_scores = np.zeros((len(opt_g), len(hole_centers)), dtype=np.float32)

                matches_g, _ = _gnn_greedy_assign(
                    opt_instances=opt_g,
                    hole_centers=hole_centers,
                    opt_hole_scores=opt_hole_scores,
                    match_radius=match_radius,
                    num_rows=s10.NUM_ROWS,
                    num_cols=s10.NUM_COLS,
                    image_h=h,
                    image_w=w,
                )
                matches_g, empty_g = apply_manual_empty_constraint(stem_base, hole_centers, matches_g, avg_spacing)
                gnn_status = 'ok'
            except Exception as ge:
                # Fallback: keep before values so summary table is always complete
                matches_g, empty_g = matches_b, empty_b
                gnn_status = 'fallback:' + str(ge).replace(',', ';')[:120]

            before_matches = len(matches_b)
            before_empty = len(empty_b)
            after_matches = len(matches_g)
            after_empty = len(empty_g)

            rows.append({
                'image_stem': stem,
                'status': gnn_status,
                'before_matches': before_matches,
                'before_empty_holes': before_empty,
                'after_matches': after_matches,
                'after_empty_holes': after_empty,
                'delta_matches': after_matches - before_matches,
                'delta_empty_holes': after_empty - before_empty,
            })
            print(f"[OK] {stem}: matches {before_matches}->{after_matches}, empty {before_empty}->{after_empty}")

        except Exception as e:
            rows.append({
                'image_stem': stem,
                'status': 'failed:' + str(e).replace(',', ';')[:180],
                'before_matches': '', 'before_empty_holes': '',
                'after_matches': '', 'after_empty_holes': '',
                'delta_matches': '', 'delta_empty_holes': '',
            })
            print(f"[FAIL] {stem}: {e}")

    # write csv
    fieldnames = [
        'image_stem', 'status',
        'before_matches', 'before_empty_holes',
        'after_matches', 'after_empty_holes',
        'delta_matches', 'delta_empty_holes'
    ]
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # overall summary
    ok_rows = [r for r in rows if r['status'] == 'ok']
    if ok_rows:
        bm = sum(int(r['before_matches']) for r in ok_rows)
        am = sum(int(r['after_matches']) for r in ok_rows)
        be = sum(int(r['before_empty_holes']) for r in ok_rows)
        ae = sum(int(r['after_empty_holes']) for r in ok_rows)
        print('--- SUMMARY ---')
        print(f"images_ok={len(ok_rows)}")
        print(f"matches_total: {bm} -> {am} (delta={am-bm})")
        print(f"empty_holes_total: {be} -> {ae} (delta={ae-be})")

    print(f"CSV saved: {out_csv}")


if __name__ == '__main__':
    main()
