import torch
from torchvision.transforms.functional import crop
import torch.nn.functional as F


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, eta=1.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            # print("x_cond shape: ", x_cond.shape, "xt shape: ", xt.shape)
            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds

def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):## i,j denote the time step parameters of the current iteration
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et_output = torch.zeros_like(x_cond, device=x.device)
            
            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                for i in range(0, len(corners), manual_batching_size):
                    outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                               xt_patch[i:i+manual_batching_size]], dim=1), t)
                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t)

            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds
def normalize_feature_maps(feature_maps):
    _, _, H, W = feature_maps.shape
    min_values, _ = torch.min(feature_maps[:, :, int(0.1 * H):int(0.9 * H), int(0.1 * W):int(0.9 * W)].contiguous().view(feature_maps.shape[0], feature_maps.shape[1], -1), dim=2)
    max_values, _ = torch.max(feature_maps[:, :, int(0.1 * H):int(0.9 * H), int(0.1 * W):int(0.9 * W)].contiguous().view(feature_maps.shape[0], feature_maps.shape[1], -1), dim=2)
    
    # Reshape the minimum and maximum values to match the original shape
    min_values = min_values.view(feature_maps.shape[0], feature_maps.shape[1], 1, 1)
    max_values = max_values.view(feature_maps.shape[0], feature_maps.shape[1], 1, 1)
    
    # Normalize the feature maps to the range of [0, 1]
    normalized_feature_maps = (feature_maps - min_values) / (max_values - min_values)        
    return normalized_feature_maps
    
def generalized_steps_multi(x, x_conds, seq, models, b=0., eta=1.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for counter, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = torch.zeros_like(x_conds[1], device=x.device)
            ets = []
            x0_list = []
            ratios = [1, 1]
            # ratios = [0, 1]
            mask = x_conds[-1]
            fea_list = []
            x_conds_input = [x_conds[0], x_conds[1]]
            masks = [mask, 1 - mask]
            for model, x_cond, ratio, mask in zip(models, x_conds_input, ratios, masks):
                et_temp, f_temp = model(torch.cat([x_cond, xt], dim=1), t)
                fea_list.append(f_temp)
                x0_temp = (xt - et_temp * (1 - at).sqrt()) / at.sqrt()
                # x0_temp = torch.clamp(x0_temp, -1, 1)
                x0_list.append(x0_temp)
                ets.append(et_temp)
                if counter < 0:
                    et += et_temp * 0.5
                else:
                    et += ratio * et_temp * mask + (1 - ratio) * et_temp * (1 - mask)
            
            if counter < 1:
                weight_A = 0.5
                weight_B = 1 - weight_A
            else:
                # x0_list[0] = torch.sigmoid(x0_list[0])
                weight_A =  torch.sigmoid(10 * (x0_list[0] - x0_list[1] - 0.))
                weight_B = 1 - weight_A#torch.abs(x0_list[1]) / (torch.abs(x0_list[1]) + torch.abs(x0_list[0]) + 1e-6)
            et_f =  weight_A * ets[0] + weight_B * ets[1]
            # et_f = torch.mean(torch.cat(ets, dim=0), dim=0)
            x0_t = (xt - et_f * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def generalized_steps_multi_weight(x, x_conds, seq, models, model_weight=None, b=0., eta=1.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        weights = []
        times = []
        edges = []
        x0_irs = []
        x0_vis = []
        xs = [x]
        weight_t = torch.ones(n).to(x.device)
        for counter, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())            
            times.append(1 * at.sqrt())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = torch.zeros_like(x_conds[1], device=x.device)
            ets = []
            x0_list = []
            for model, x_cond in zip(models, x_conds):
                et_temp = model(torch.cat([x_cond, xt], dim=1), t)
                x0_temp = (xt - et_temp * (1 - at).sqrt()) / at.sqrt()
                # x0_temp.clamp_(-1., 1.)
                x0_list.append(x0_temp)    
                ets.append(et_temp)
            # model_weight = None
            if model_weight is not None:
                if counter == 0: #If it is noise from the first round of estimation, fused using preset weights [0.5, 0,5]
                    weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] ## The channel dimension is 1
                    weight_B = 1 - weight_A
                    weights.append(weight_A)
                else:
                    if counter == 1: ## The main consideration is that the noise estimated in the first step is not accurate enough
                        x0_ir = x0_list[0]
                        # weight_input = torch.cat([x0_ir, x0_ir, tried_weight], dim=1)
                        weight_input = torch.cat([x0_ir, x0_list[0], weights[-1]], dim=1)
                    elif counter < (len(seq) - 1): 
                        last_x0_ir = x0_list[0]
                        weight_input = torch.cat([x0_ir, x0_list[0], weights[-1]], dim=1)
                    else:
                        weight_input = torch.cat([x0_ir, last_x0_ir, weights[-1]], dim=1) ## 
                    weight_A, edge = model_weight(weight_input, weight_t)
                    # weight_A, edge = model_weight(weight_input, t)
                    weight_B  = 1 - weight_A      
                    weights.append(weight_A)
                    edges.append(edge)
            else:
                weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] ## The channel dimension is 1
                weight_B = 1 - weight_A
                weights.append(weight_A)
            x0_irs.append(x0_list[0])
            x0_vis.append(x0_list[1])
            et_f =  weight_A * ets[0] + weight_B * ets[1]
            x0_t = (xt - et_f * (1 - at).sqrt()) / at.sqrt()
            
            x0_preds.append(x0_t)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            
            xs.append(xt_next.to('cpu'))
            results = {
                'xs':xs, 
                'x0':x0_preds,
                'weight':weights,
                'edge':edges,
                'x0_ir':x0_irs,
                'x0_vi':x0_vis,
                'time':times
            }
    return results

def generalized_steps_multi_weight_train(x, x_conds, seq, models, model_weight=None, b=0., eta=1.):    
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    weights = []
    times = []
    xs = [x]
    x0_irs = []
    x0_vis = []
    edges = []
    weight_t = torch.ones(n).to(x.device) ##
    for counter, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        
        times.append(1 * at.sqrt())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        et = torch.zeros_like(x_conds[1], device=x.device)
        ets = []
        x0_list = []
        for model, x_cond in zip(models, x_conds):
            et_temp = model(torch.cat([x_cond, xt], dim=1), t)
            x0_temp = (xt - et_temp * (1 - at).sqrt()) / at.sqrt() 
            # x0_temp.clamp_(-1., 1.)
            x0_list.append(x0_temp)    
            ets.append(et_temp)
        if model_weight is not None:
            if counter == 0: # If the noise is from the first round of estimation, it is fused using the preset weights [0.5, 0,5]
                weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] ## channel维度是1
                weight_B = 1 - weight_A
                weights.append(weight_A)
            else:
                if counter == 1: ## The main consideration is that the noise estimated in the first step is not accurate enough
                    x0_ir = x0_list[0]
                    # weight_input = torch.cat([x0_ir, x0_ir, tried_weight], dim=1)
                    weight_input = torch.cat([x0_ir, x0_list[0], weights[-1]], dim=1)
                elif counter < (len(seq) - 1): 
                    last_x0_ir = x0_list[0]
                    weight_input = torch.cat([x0_ir, x0_list[0], weights[-1]], dim=1)
                else:
                    weight_input = torch.cat([x0_ir, last_x0_ir, weights[-1]], dim=1) 
                weight_A, edge = model_weight(weight_input, weight_t)
                weight_B  = 1 - weight_A      
                weights.append(weight_A)
                edges.append(edge)
        else:
            weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] ## The channel dimension is 1
            weight_B = 1 - weight_A
            weights.append(weight_A)
        x0_irs.append(x0_list[0])
        x0_vis.append(x0_list[1])
        et_f =  weight_A * ets[0] + weight_B * ets[1]
        # et_f = torch.mean(torch.cat(ets, dim=0), dim=0)
        x0_t = (xt - et_f * (1 - at).sqrt()) / at.sqrt()        
        # x0_t.clamp_(-1., 1.)
        x0_preds.append(x0_t)
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        
        # xt_next.clamp_(-1., 1.)
        xs.append(xt_next.to('cpu'))
        results = {
            'xs':xs, 
            'x0':x0_preds,
            'weight':weights,
            'edge':edges,
            'x0_ir':x0_irs,
            'x0_vi':x0_vis,
            'time':times
        }
    return results

def generalized_steps_mif(x, x_conds, seq, models, model_weight=None, b=0., eta=1.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        weights = []
        times = []
        edges = []
        x0_irs = []
        x0_vis = []
        xs = [x]
        weight_t = torch.ones(n).to(x.device) 
        for counter, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())            
            times.append(1 * at.sqrt())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = torch.zeros_like(x_conds[1], device=x.device)
            ets = []
            x0_list = []
            for model, x_cond in zip(models, x_conds):
                et_temp = model(torch.cat([x_cond, xt], dim=1), t)
                x0_temp = (xt - et_temp * (1 - at).sqrt()) / at.sqrt()
                # x0_temp.clamp_(-1., 1.)
                x0_list.append(x0_temp)    
                ets.append(et_temp)
            # model_weight = None
            if model_weight is not None:
                if counter == 0: 
                    weight_A =  -1.0 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::]
                    weight_B = 1 - weight_A
                    weights.append(weight_A)
                    weight_input = torch.cat([x0_list[0], x0_list[0], weights[-1]], dim=1)
                
                ### Here is the code to calculate the dynamic weights
                weight_A, edge = model_weight(weight_input, weight_t)
                # weight_A, edge = model_weight(weight_input, t)
                weight_B  = 1 - weight_A      
                weights.append(weight_A)
                edges.append(edge)
            else:
                weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] 
                weight_B = 1 - weight_A
                weights.append(weight_A)
            x0_irs.append(x0_list[0])
            x0_vis.append(x0_list[1])
            et_f =  weight_A * ets[0] + weight_B * ets[1]
            # et_f = torch.mean(torch.cat(ets, dim=0), dim=0)
            x0_t = (xt - et_f * (1 - at).sqrt()) / at.sqrt()
            
            # x0_t.clamp_(-1., 1.)
            x0_preds.append(x0_t)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            # xt_next.clamp_(-1., 1.)
            xs.append(xt_next.to('cpu'))
            results = {
                'xs':xs, 
                'x0':x0_preds,
                'weight':weights,
                'edge':edges,
                'x0_ir':x0_irs,
                'x0_vi':x0_vis,
                'time':times
            }
    return results


def generalized_steps_mif_train(x, x_conds, seq, models, model_weight=None, b=0., eta=1.):    
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    weights = []
    weights_B = []
    times = []
    xs = [x]
    x0_irs = []
    x0_vis = []
    edges = []
    weight_t = torch.ones(n).to(x.device)  
    for counter, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        
        times.append(1 * at.sqrt())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        et = torch.zeros_like(x_conds[1], device=x.device)
        ets = []
        x0_list = [] 
        for model, x_cond in zip(models, x_conds):
            et_temp = model(torch.cat([x_cond, xt], dim=1), t)
            x0_temp = (xt - et_temp * (1 - at).sqrt()) / at.sqrt() 
            
            x0_list.append(x0_temp)    
            ets.append(et_temp)
        if model_weight is not None:
            if counter == 0: 
                weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] ## channel维度是1
                weight_B = 1 - weight_A
                weights.append(weight_A)
                weights_B.append(weight_B)
            else:
                if counter == 1:
                    x0_ir = x0_list[0]
                    # weight_input = torch.cat([x0_ir, x0_ir, tried_weight], dim=1)
                    weight_input_A = torch.cat([xt, x0_list[0], weights[-1]], dim=1)
                    weight_input_B = torch.cat([xt, x0_list[1], weights_B[-1]], dim=1)
                elif counter < (len(seq) - 1): 
                    last_x0_A = x0_list[0]
                    last_x0_B = x0_list[1]
                    weight_input_A = torch.cat([xt, x0_list[0], weights[-1]], dim=1)
                    weight_input_B = torch.cat([xt, x0_list[1], weights_B[-1]], dim=1)
                else:
                    weight_input_A = torch.cat([xt, last_x0_A, weights[-1]], dim=1) 
                    weight_input_B = torch.cat([xt, last_x0_B, weights_B[-1]], dim=1) 
                weight_A, edge = model_weight(weight_input_A, weight_t)
                weight_B, edge = model_weight(weight_input_B, weight_t)
                concat_weights = torch.cat([weight_A, weight_B], dim=1)
                softmax_weights = F.softmax(input=concat_weights, dim=1)
                weight_A = softmax_weights[:, 0, ::].unsqueeze(1)
                weight_B = softmax_weights[:, 1, ::].unsqueeze(1)
                weights.append(weight_A)
                weights_B.append(weight_B)
                edges.append(edge)
        else:
            weight_A =  0.5 * torch.ones_like(x_conds[1], device=x.device)[:, :1, ::] 
            weight_B = 1 - weight_A
            weights.append(weight_A)
            weights_B.append(weight_B)
        x0_irs.append(x0_list[0])
        x0_vis.append(x0_list[1])
        et_f =  weight_A * ets[0] + weight_B * ets[1]
        # et_f = torch.mean(torch.cat(ets, dim=0), dim=0)
        x0_t = (xt - et_f * (1 - at).sqrt()) / at.sqrt()        
        # x0_t.clamp_(-1., 1.)
        x0_preds.append(x0_t)
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        
        # xt_next.clamp_(-1., 1.)
        xs.append(xt_next.to('cpu'))
        results = {
            'xs':xs, 
            'x0':x0_preds,
            'weight':weights,
            'edge':edges,
            'x0_ir':x0_irs,
            'x0_vi':x0_vis,
            'time':times
        }
    return results