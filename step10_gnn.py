#!/usr/bin/env python3
# GNN helpers for step10_hole.py - pure PyTorch GAT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

NUM_ROWS=20; NUM_COLS=10
SAME_HOLE_CENTER_DIST_MAX=60.0; OVERLAP_IOU_MIN=0.08
OVERLAP_DILATED_IOU_MIN=0.18; MASK_DILATE_KERNEL=5; MASK_DILATE_ITERS=1
SMALL_FRAGMENT_AREA_RATIO=0.45

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4):
        super().__init__()
        self.heads=heads; self.head_dim=out_dim//heads; self.out_dim=out_dim
        self.W_node=nn.Linear(in_dim,out_dim,bias=False)
        self.W_edge=nn.Linear(edge_dim,heads,bias=False)
        self.att_src=nn.Parameter(torch.empty(1,heads,self.head_dim))
        self.att_dst=nn.Parameter(torch.empty(1,heads,self.head_dim))
        self.bias=nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.att_src); nn.init.xavier_uniform_(self.att_dst)
    def forward(self,x,ei,ea):
        N=x.size(0); Wh=self.W_node(x).view(N,self.heads,self.head_dim)
        s,d=ei[0],ei[1]
        e=F.leaky_relu((Wh[s]*self.att_src).sum(-1)+(Wh[d]*self.att_dst).sum(-1)+self.W_edge(ea),0.2)
        ex=torch.exp(e-e.max()); den=torch.zeros(N,self.heads,device=x.device)
        den.scatter_add_(0,d.unsqueeze(1).expand(-1,self.heads),ex)
        alpha=ex/(den[d]+1e-9)
        out=torch.zeros(N,self.heads,self.head_dim,device=x.device)
        out.scatter_add_(0,d.unsqueeze(1).unsqueeze(2).expand(-1,self.heads,self.head_dim),Wh[s]*alpha.unsqueeze(-1))
        return F.elu(out.view(N,self.out_dim)+self.bias)

class InstanceGNN(nn.Module):
    ND=7; ED=6; HD=4; HID=64
    def __init__(self):
        super().__init__()
        self.g1=GATLayer(self.ND,self.HID,self.ED)
        self.g2=GATLayer(self.HID,self.HID,self.ED)
        self.mh=nn.Sequential(nn.Linear(self.HID*2+self.ED,32),nn.ReLU(),nn.Linear(32,1))
        self.hh=nn.Sequential(nn.Linear(self.HID+self.HD,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,nf,ei,ef,mp,mef,hf):
        h=self.g2(self.g1(nf,ei,ef),ei,ef)
        ml=self.mh(torch.cat([h[mp[0]],h[mp[1]],mef],-1)).squeeze(-1) if mp.size(1)>0 else torch.zeros(0,device=nf.device)
        Ni,Nh=h.size(0),hf.size(0)
        hl=self.hh(torch.cat([h.unsqueeze(1).expand(-1,Nh,-1),hf.unsqueeze(0).expand(Ni,-1,-1)],-1)).squeeze(-1)
        return ml,hl

class _UF:
    def __init__(self,n): self.p=list(range(n)); self.r=[0]*n
    def find(self,x):
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: self.p[ra]=rb
        elif self.r[ra]>self.r[rb]: self.p[rb]=ra
        else: self.p[rb]=ra; self.r[ra]+=1

def _iou(m1,m2):
    i=np.logical_and(m1>0,m2>0).sum(); u=np.logical_or(m1>0,m2>0).sum()
    return float(i/u) if u>0 else 0.0
def _diou(m1,m2):
    k=np.ones((MASK_DILATE_KERNEL,MASK_DILATE_KERNEL),np.uint8)
    return _iou(cv2.dilate(m1.astype(np.uint8),k,iterations=MASK_DILATE_ITERS),
                cv2.dilate(m2.astype(np.uint8),k,iterations=MASK_DILATE_ITERS))
def _touch(b1,b2,m=6):
    ax1,ay1,ax2,ay2=b1; bx1,by1,bx2,by2=b2
    return not(ax2<bx1-m or bx2<ax1-m or ay2<by1-m or by2<ay1-m)

def build_gnn_inputs(instances,masks,har,adj,dmat,holes,ih,iw,device):
    N=len(instances)
    cx=np.array([i["centroid"][0] for i in instances],np.float32)
    cy=np.array([i["centroid"][1] for i in instances],np.float32)
    ar=np.array([i["area"] for i in instances],np.float32)
    bb=np.array([i["bbox"] for i in instances],np.float32)
    hp=np.array([h["center"] for h in holes],np.float32)
    ip=np.stack([cx,cy],1)
    mhd=np.linalg.norm(ip[:,None,:]-hp[None,:,:],axis=2).min(1)/(max(iw,ih)+1e-6)
    ise=np.zeros(N,np.float32)
    for idx,row in enumerate(har):
        hi=row["best_hole_index"]
        if hi is not None:
            hc=holes[hi]
            if hc["row"] in [0,NUM_ROWS-1] or hc["col"] in [0,NUM_COLS-1]: ise[idx]=1.0
    nf=np.stack([cx/(iw+1e-6),cy/(ih+1e-6),np.log1p(ar)/15.0,
                 bb[:,2]/(iw+1e-6),bb[:,3]/(ih+1e-6),mhd,ise],1)
    ii,jj=np.where(np.triu(adj>0,k=1))
    bbl=[i["bbox"] for i in instances]
    es,ed,ef=[],[],[]; mp,mef,ps=[],[],[]
    for i,j in zip(ii.tolist(),jj.tolist()):
        dij=float(dmat[i,j])
        hi2=har[i]["best_hole_index"]; hj2=har[j]["best_hole_index"]
        sh=float(hi2 is not None and hj2 is not None and hi2==hj2)
        iou=diou=0.0
        if _touch(bbl[i],bbl[j]): iou=_iou(masks[i],masks[j]); diou=_diou(masks[i],masks[j])
        ai_=float(instances[i]["area"]); aj_=float(instances[j]["area"])
        arat=min(ai_,aj_)/max(max(ai_,aj_),1e-6)
        dn=dij/(max(iw,ih)+1e-6); aw=float(adj[i,j])
        e=[dn,iou,diou,arat,sh,aw]
        es+=[i,j]; ed+=[j,i]; ef+=[e,e]; mp.append([i,j]); mef.append(e)
        sm=False
        if sh:
            if dij<=SAME_HOLE_CENTER_DIST_MAX: sm=True
            if iou>=OVERLAP_IOU_MIN or diou>=OVERLAP_DILATED_IOU_MIN: sm=True
            if arat<=SMALL_FRAGMENT_AREA_RATIO and dij<=SAME_HOLE_CENTER_DIST_MAX: sm=True
        if hi2 is None and hj2 is None:
            if dij<=25.0 and (iou>0 or diou>=0.22): sm=True
        ps.append(float(sm))
    ei=torch.tensor([es,ed],dtype=torch.long,device=device) if es else torch.zeros((2,0),dtype=torch.long,device=device)
    eft=torch.tensor(ef,dtype=torch.float32,device=device) if ef else torch.zeros((0,6),dtype=torch.float32,device=device)
    mpt=torch.tensor(mp,dtype=torch.long,device=device).T if mp else torch.zeros((2,0),dtype=torch.long,device=device)
    meft=torch.tensor(mef,dtype=torch.float32,device=device) if mef else torch.zeros((0,6),dtype=torch.float32,device=device)
    pst=torch.tensor(ps,dtype=torch.float32,device=device) if ps else torch.zeros(0,dtype=torch.float32,device=device)
    hfeat=np.array([[h["center"][0]/(iw+1e-6),h["center"][1]/(ih+1e-6),
                     h["row"]/(NUM_ROWS+1e-6),h["col"]/(NUM_COLS+1e-6)] for h in holes],np.float32)
    return (torch.tensor(nf,dtype=torch.float32,device=device),
            ei,eft,mpt,meft,torch.tensor(hfeat,dtype=torch.float32,device=device),pst)

def gnn_merge_and_assign(instances,masks,har,adj,dmat,holes,ih,iw,device,n_iter=40):
    if not instances: return {},np.zeros((0,len(holes)))
    nf,ei,ef,mp,mef,hf,ps=build_gnn_inputs(instances,masks,har,adj,dmat,holes,ih,iw,device)
    gnn=InstanceGNN().to(device)
    opt=torch.optim.Adam(gnn.parameters(),lr=5e-3,weight_decay=1e-4)
    for _ in range(n_iter):
        gnn.train(); opt.zero_grad()
        ml,hl=gnn(nf,ei,ef,mp,mef,hf)
        loss=torch.tensor(0.0,device=device)
        if ml.numel()>0: loss=loss+F.binary_cross_entropy_with_logits(ml,ps)
        bhi=torch.tensor([r["best_hole_index"] if r["best_hole_index"] is not None else 0
                          for r in har],dtype=torch.long,device=device)
        hasm=torch.tensor([1.0 if r["best_hole_index"] is not None else 0.0
                           for r in har],dtype=torch.float32,device=device)
        loss=loss+0.5*(F.cross_entropy(hl,bhi,reduction='none')*hasm).mean()
        loss.backward(); opt.step()
    gnn.eval()
    with torch.no_grad(): ml,hl=gnn(nf,ei,ef,mp,mef,hf)
    n=len(instances); uf=_UF(n)
    if mp.size(1)>0:
        prob=torch.sigmoid(ml).cpu().numpy()
        for k in range(mp.size(1)):
            i,j=int(mp[0,k]),int(mp[1,k])
            if prob[k]>0.5 or ps[k].item()>0.5: uf.union(i,j)
    groups={}
    for i in range(n): groups.setdefault(uf.find(i),[]).append(i)
    return groups,hl.cpu().numpy()


def _gnn_greedy_assign(opt_instances, hole_centers, opt_hole_scores,
                       match_radius, num_rows, num_cols, image_h, image_w):
    """
    Greedy bipartite matching using GNN hole scores.
    opt_hole_scores: (N_opt, N_hole) numpy array
    Returns: matches, empty_holes (same format as greedy_match_optimized_instances_with_mask_overlap)
    """
    N_opt  = len(opt_instances)
    N_hole = len(hole_centers)
    if N_opt == 0 or N_hole == 0:
        empty_holes = [{'hole_index':int(h['hole_index']),'row':int(h['row']),'col':int(h['col']),
                        'theory_center_x':float(h['center'][0]),'theory_center_y':float(h['center'][1])}
                       for h in hole_centers]
        return [], empty_holes

    # build candidate (score, oi, hi) list with distance gate
    candidates = []
    for oi, ins in enumerate(opt_instances):
        cx, cy = float(ins['centroid'][0]), float(ins['centroid'][1])
        for hi, h in enumerate(hole_centers):
            hcx, hcy = float(h['center'][0]), float(h['center'][1])
            dist = float(np.hypot(cx - hcx, cy - hcy))
            radius = match_radius
            if h['row'] in [0, num_rows-1] or h['col'] in [0, num_cols-1]:
                radius = min(radius + 5, 90)
            if dist > radius * 1.5:  # loose gate: allow GNN to extend reach
                continue
            score = float(opt_hole_scores[oi, hi])
            candidates.append((score, oi, hi, dist))

    candidates.sort(key=lambda x: -x[0])
    used_holes = set(); used_insts = set(); matches = []
    for score, oi, hi, dist in candidates:
        if oi in used_insts or hi in used_holes: continue
        used_insts.add(oi); used_holes.add(hi)
        h = hole_centers[hi]; ins = opt_instances[oi]
        matches.append({
            'hole_index': int(h['hole_index']),
            'row': int(h['row']), 'col': int(h['col']),
            'theory_center_x': float(h['center'][0]),
            'theory_center_y': float(h['center'][1]),
            'opt_instance_index': int(ins['opt_instance_index']),
            'instance_centroid_x': float(ins['centroid'][0]),
            'instance_centroid_y': float(ins['centroid'][1]),
            'distance': dist,
            'match_score': score,
            'source_instance_indices': ','.join(map(str, ins['source_instance_indices']))
        })

    matched_holes = {m['hole_index'] for m in matches}
    empty_holes = []
    for h in hole_centers:
        if int(h['hole_index']) not in matched_holes:
            empty_holes.append({
                'hole_index': int(h['hole_index']),
                'row': int(h['row']), 'col': int(h['col']),
                'theory_center_x': float(h['center'][0]),
                'theory_center_y': float(h['center'][1]),
            })
    return matches, empty_holes
