"""FCOS-style center sampling and scale assignment for our one-class detector."""
import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss,sigmoid_focal_loss
from src.architectures import flatten_outputs,distances_to_boxes

def detection_loss(outputs,targets):
    logits,pred,centers,points,strides=flatten_outputs(outputs)
    total=logits.sum()*0
    for batch,boxes in enumerate(targets):
        n=len(points)
        assigned=torch.zeros(n,device=points.device)
        target_dist=torch.zeros((n,4),device=points.device)
        positive=torch.zeros(n,dtype=torch.bool,device=points.device)
        if len(boxes):
            lt=points[:,None,:]-boxes[None,:,:2]
            rb=boxes[None,:,2:]-points[:,None,:]
            distances=torch.cat((lt,rb),2)
            inside=distances.min(2).values>0
            centers_gt=(boxes[:,:2]+boxes[:,2:])/2
            near=(points[:,None,:]-centers_gt[None]).abs().max(2).values <= strides[:,None]*1.5
            size=distances.max(2).values
            lower=torch.where(strides==8,0,torch.where(strides==16,64,128))[:,None]
            upper=torch.where(strides==8,64,torch.where(strides==16,128,1e8))[:,None]
            eligible=inside & near & (size>=lower) & (size<upper)
            areas=((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))[None].expand(n,-1).clone()
            areas[~eligible]=float("inf")
            values,indices=areas.min(1)
            positive=torch.isfinite(values)
            target_dist=distances[torch.arange(n,device=points.device),indices]
            assigned[positive]=1
        norm=max(1,int(positive.sum()))
        loss_cls=sigmoid_focal_loss(logits[batch],assigned,alpha=0.25,gamma=2,reduction="sum")/norm
        total=total+loss_cls
        if positive.any():
            d=target_dist[positive]
            lr=d[:,[0,2]]
            tb=d[:,[1,3]]
            ctr=((lr.min(1).values/lr.max(1).values)*(tb.min(1).values/tb.max(1).values)).sqrt()
            a=distances_to_boxes(points[positive],pred[batch,positive])
            b=distances_to_boxes(points[positive],d)
            box_loss=generalized_box_iou_loss(a,b,reduction="none")
            total=total+(box_loss*ctr).sum()/ctr.sum().clamp(min=1e-6)
            total=total+F.binary_cross_entropy_with_logits(centers[batch,positive],ctr)
    return total/len(targets)
