import numpy as np

def _gnn_greedy_assign(opt_instances, hole_centers, opt_hole_scores,
                       match_radius, num_rows, num_cols, image_h, image_w):
    N_opt = len(opt_instances)
    N_hole = len(hole_centers)
    if N_opt == 0 or N_hole == 0:
        return [], [{'hole_index':int(h['hole_index']),'row':int(h['row']),'col':int(h['col']),
                     'theory_center_x':float(h['center'][0]),'theory_center_y':float(h['center'][1])}
                    for h in hole_centers]
    candidates = []
    for oi, ins in enumerate(opt_instances):
        cx, cy = float(ins['centroid'][0]), float(ins['centroid'][1])
        for hi, h in enumerate(hole_centers):
            hcx, hcy = float(h['center'][0]), float(h['center'][1])
            dist = float(np.hypot(cx-hcx, cy-hcy))
            rad = min(match_radius+5, 90) if h['row'] in [0,num_rows-1] or h['col'] in [0,num_cols-1] else match_radius
            if dist > rad * 1.5: continue
            candidates.append((float(opt_hole_scores[oi, hi]), oi, hi, dist))
    candidates.sort(key=lambda x: -x[0])
    used_h = set(); used_i = set(); matches = []
    for score, oi, hi, dist in candidates:
        if oi in used_i or hi in used_h: continue
        used_i.add(oi); used_h.add(hi)
        h = hole_centers[hi]; ins = opt_instances[oi]
        matches.append({'hole_index':int(h['hole_index']),'row':int(h['row']),'col':int(h['col']),
            'theory_center_x':float(h['center'][0]),'theory_center_y':float(h['center'][1]),
            'opt_instance_index':int(ins['opt_instance_index']),
            'instance_centroid_x':float(ins['centroid'][0]),'instance_centroid_y':float(ins['centroid'][1]),
            'distance':dist,'match_score':score,
            'source_instance_indices':','.join(map(str,ins['source_instance_indices']))})
    matched = {m['hole_index'] for m in matches}
    empty = [{'hole_index':int(h['hole_index']),'row':int(h['row']),'col':int(h['col']),
               'theory_center_x':float(h['center'][0]),'theory_center_y':float(h['center'][1])}
              for h in hole_centers if int(h['hole_index']) not in matched]
    return matches, empty
