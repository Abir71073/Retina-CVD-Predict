"""Shared implementation of the project's existing U-Net patch/width workflow."""
import cv2
import numpy as np


def segment_vessels(model, rgb, stride=128):
    patch = int(model.input_shape[1])
    if model.input_shape[2] != patch or not 0 < stride <= patch:
        raise ValueError('Expected square model patches and 0 < stride <= patch size.')
    h,w=rgb.shape[:2]
    # Cover every border even when dimensions are smaller than one patch.
    padded_h=max(patch, patch+int(np.ceil(max(0,h-patch)/stride))*stride)
    padded_w=max(patch, patch+int(np.ceil(max(0,w-patch)/stride))*stride)
    image=np.pad(rgb.astype(np.float32)/255.,((0,padded_h-h),(0,padded_w-w),(0,0)))
    accum=np.zeros((padded_h,padded_w),np.float32);weights=np.zeros_like(accum)
    x,y=np.meshgrid(np.linspace(-1,1,patch),np.linspace(-1,1,patch))
    blend=np.exp(-(x*x+y*y)/2).astype(np.float32)
    for yy in range(0,padded_h-patch+1,stride):
        for xx in range(0,padded_w-patch+1,stride):
            batch=image[None,yy:yy+patch,xx:xx+patch]
            inputs=[batch] if isinstance(getattr(model,'input',None),(list,tuple)) else batch
            pred=np.asarray(model(inputs,training=False))
            if pred.shape!=(1,patch,patch,1) or not np.isfinite(pred).all():
                raise ValueError('Invalid vessel segmentation output.')
            accum[yy:yy+patch,xx:xx+patch]+=pred[0,:,:,0]*blend
            weights[yy:yy+patch,xx:xx+patch]+=blend
    return (accum/weights)[:h,:w]


def measure_width_proxy(rgb, mask, method='halves'):
    """Retain the original width heuristic; it does not identify artery/vein classes."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(51,51),0)
    _,bright=cv2.threshold(blur,max(0,float(blur.max())-15),255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {'status':'unavailable','reason':'No candidate optic-disc region.'}
    (cx,cy),radius=cv2.minEnclosingCircle(max(contours,key=cv2.contourArea))
    if radius<2 or radius>min(gray.shape)/3:
        return {'status':'unavailable','reason':'Implausible optic-disc candidate.'}
    center=(int(cx),int(cy));zone=np.zeros(gray.shape,np.uint8)
    cv2.circle(zone,center,int(radius*3),1,-1);cv2.circle(zone,center,int(radius*2),0,-1)
    vessels=(mask>.5).astype(np.uint8)*zone
    distance=cv2.distanceTransform(vessels,cv2.DIST_L2,3)
    skeleton=cv2.ximgproc.thinning(vessels*255)
    widths=distance[skeleton==255]*2
    if method=='kmeans': widths=widths[widths>=3]
    if len(widths)<10: return {'status':'unavailable','reason':'Too few vessel samples in the measurement ring.'}
    if method=='kmeans':
        cv2.setRNGSeed(42)
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.)
        _,_,centers=cv2.kmeans(np.float32(widths).reshape(-1,1),2,None,criteria,10,cv2.KMEANS_PP_CENTERS)
        narrow,wide=sorted(centers[:,0])
    elif method=='halves':
        widths=np.sort(widths);mid=len(widths)//2
        narrow=float(np.mean(widths[:mid][-10:]));wide=float(np.mean(widths[mid:][-10:]))
    else: raise ValueError('Unknown width measurement method.')
    if wide<=0: return {'status':'unavailable','reason':'Zero vessel width.'}
    return {'status':'available','width_ratio_proxy':float(narrow/wide),
            'narrow_width_pixels':float(narrow),'wide_width_pixels':float(wide),
            'center':center,'radius':float(radius),'samples':int(len(widths)),
            'method':method,'note':'Width groups are not artery/vein identification or a calibrated AVR diagnosis.'}


def retinal_field_mask(rgb):
    """Estimate the visible field for image-space measurements, not a fundus gate."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    candidate = np.uint8(gray > 8) * 255
    size = max(3, (min(gray.shape) // 100) | 1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
    return mask.astype(bool)
