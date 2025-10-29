import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    rank = dist.get_rank()
    device = torch.device('cuda', rank)

    n_examples = torch.zeros(
        (), dtype=torch.int, device=device, requires_grad=False)
    n_correct = torch.zeros((), dtype=torch.int,
                            device=device, requires_grad=False)
    n_batches = torch.zeros((), dtype=torch.int,
                            device=device, requires_grad=False)
    total_cross_entropy = torch.zeros((), device=device, requires_grad=False)
    total_cluster_cost = torch.zeros((), device=device, requires_grad=False)
    # separation cost is meaningful only for class_specific
    total_separation_cost = torch.zeros((), device=device, requires_grad=False)
    total_avg_separation_cost = torch.zeros(
        (), device=device, requires_grad=False)

    # SPRNet
    total_global_clip_cost = torch.zeros(
        (), device=device, requires_grad=False)
    total_local_clip_cost = torch.zeros((), device=device, requires_grad=False)

    with torch.no_grad():
        # global_attribute_features [112, 1024]
        global_attribute_features = model.module.text_embed(
            dataloader.dataset.global_attribute_tokens, device=device)
        global_attribute_features = global_attribute_features / \
            global_attribute_features.norm(dim=1, keepdim=True)
        # global_attribute_features [112, 1024]

    pbar = tqdm(dataloader, desc=f'[r{rank}] ', position=rank)
    for i, (image, label, attribute, global_str) in enumerate(pbar):
        input = image.to(device, non_blocking=True)
        target = label.to(device, non_blocking=True)
        # all global_attributes are same
        # global_attributes = [x[0] for x in global_attributes]

        # SPRNet
        attribute = attribute.to(device)
        # tokenize is in the text_embed function
        # caption_text = model.tokenize(global_str).to(device)
        # attribute_text = model.tokenize(global_attributes).to(device)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # SPRNet
            output, min_distances, global_features, local_features = model.module.image_embed(
                input)
            text_features = model.module.text_embed(global_str, device=device)
            # text_features [N, 1024]
            # global_attribute_features = model.module.text_embed(
            #     global_attributes, device=device)
            # global_attribute_features [112, 1024]

            # normalized features
            image_features = global_features / \
                global_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            # TODO: logit_scale required?
            # logit_scale = model.clip.logit_scale.exp()
            logit_scale = 100
            logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(input.size(
                0), dtype=torch.long, device=device)

            # TODO: local -> nan ?
            local_features = local_features / \
                local_features.norm(dim=1, keepdim=True)
            # local_features [N, 1024, 7, 7]

            # logit_scale added
            local_logits = logit_scale * torch.einsum(
                "ijkl,mj->iklm", local_features, global_attribute_features
            )
            # local_logits [N, 7, 7, 112]

            # TODO: use soft max instead of amax
            max_local_logits = torch.amax(local_logits, dim=(1, 2))
            # max_local_logits [N, 112]

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(
                    model.module.prototype_class_identity[:, label]).to(device)
                inverted_distances, _ = torch.max(
                    (max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) *
                              prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(
                    max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class,
                              dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - \
                        torch.t(model.module.prototype_class_identity).to(
                            device)
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # clip loss
            global_clip_cost = (
                F.cross_entropy(logits_per_image, ground_truth)
                + F.cross_entropy(logits_per_text, ground_truth)
            ) / 2

            # TODO: local -> nan ?
            # local_clip_cost = torch.sum(
            #     min_local_distances * attribute, dim=1
            # ) - torch.sum(min_local_distances * (1 - attribute), dim=1)
            # local_clip_cost = torch.mean(local_clip_cost)

            # new multilabel contrastive loss
            eps = 1e-12
            local_clip_cost = torch.mean(-(1 / torch.sum(attribute, dim=1)) * torch.sum(attribute * (max_local_logits - torch.log(
                torch.sum((1 - attribute) * torch.exp(max_local_logits), dim=1, keepdim=True) + eps)), dim=1))

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1
                            + coefs['global_clip'] * global_clip_cost
                            + coefs['local_clip'] * local_clip_cost)
                else:
                    # loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
                    raise
            else:
                '''
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
                '''
                raise

            # Reduce the loss across GPUs
            dist.all_reduce(loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pbar.set_postfix({
            #     'loss': loss.item(),
            # })

        # evaluation statistics
        with torch.no_grad():
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum()

            n_batches += 1
            total_cross_entropy += cross_entropy
            total_cluster_cost += cluster_cost
            total_separation_cost += separation_cost
            total_avg_separation_cost += avg_separation_cost
            total_global_clip_cost += global_clip_cost
            total_local_clip_cost += local_clip_cost

        # del input
        # del target
        # del output
        # del predicted
        # del min_distances

    end = time.time()

    dist.all_reduce(n_examples)
    dist.all_reduce(n_correct)
    dist.all_reduce(n_batches)
    dist.all_reduce(total_cross_entropy)
    dist.all_reduce(total_cluster_cost)
    dist.all_reduce(total_separation_cost)
    dist.all_reduce(total_avg_separation_cost)
    dist.all_reduce(total_global_clip_cost)
    dist.all_reduce(total_local_clip_cost)

    if rank == 0:
        log('\ttime: \t{0}'.format(end - start))
        log('\tcross ent: \t{0}'.format(
            total_cross_entropy.item() / n_batches.item()))
        log('\tcluster: \t{0}'.format(
            total_cluster_cost.item() / n_batches.item()))
        if class_specific:
            log('\tseparation:\t{0}'.format(
                total_separation_cost.item() / n_batches.item()))
            log('\tavg separation:\t{0}'.format(
                total_avg_separation_cost.item() / n_batches.item()))
        log('\tglobal clip: \t{0}'.format(
            total_global_clip_cost.item() / n_batches.item()))
        log('\tlocal clip: \t{0}'.format(
            total_local_clip_cost.item() / n_batches.item()))
        log('\taccu: \t\t{0}%'.format(
            n_correct.item() / n_examples.item() * 100))
        log('\tl1: \t\t{0}'.format(
            model.module.last_layer.weight.norm(p=1).item()))

        # warning: out of memory
        # p = model.module.prototype_vectors.view(
        #     model.module.num_prototypes, -1).cpu()
        # with torch.no_grad():
        #     p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        # log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct.item() / n_examples.item()


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert (optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.image_encoder.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.text_encoder.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.image_encoder.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.text_encoder.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = False

    log('\twarm')


def joint(model, log=print):
    for p in model.module.image_encoder.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.text_encoder.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = False

    log('\tjoint')
