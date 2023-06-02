import torch
from torch import nn, optim
import copy

# learn how to learn trajectory while retaining the global distribution
# global distribution := p(r|t)
# learned distribution := p(r_2|t, r_1)

# 1. input a trajectory to the generator (e.g., [(r_1,t_1), (r_2,t_2), (r_3,t_3)])
# 2. learn genrator((r_1,t_1))=(r_2,t_2) and generator((r_2,t_2))=(r_3,t_3), which results in generator'
# 3. input a trajectory to the generator' (e.g., [(r_1,t_1), (r_2,t_2), (r_3,t_3)])
# 4. learn generator'((r_1,t_1))=p(r|t_1) and generator'((r_2,t_2))=p(r|t_2)

def time_to_index(time, time_ranges):
    for i in range(len(time_ranges)):
        if time <= time_ranges[i][0]:
            return i
    return len(time_ranges)-1

def meta_learning(data_loader, temp_generator, generator, loss_model, loss_time_fn, ignore_index, lr, loss_kl, global_distributions, noise_scale, time_ranges, n_iter):
    cuda_number = next(temp_generator.parameters()).device.index
    # move the generator to cpu
    generator = generator.cpu()
    # deepcopy the parameters of generator._module to temp_generator
    temp_generator.load_state_dict(generator._module.state_dict())
    # move the temp_generator to cuda
    temp_generator = temp_generator.cuda(cuda_number)

    meta_optim = optim.Adam(temp_generator.parameters(), lr=lr)

    count = 0


    for batch in data_loader:
        
        count += 1
        input_locations = batch["input"].cuda(cuda_number, non_blocking=True)

        batch_size = input_locations.shape[0]
        seq_len = input_locations.shape[1]

        target_locations = batch["target"].cuda(cuda_number, non_blocking=True)
        labels = batch["label"].cuda(cuda_number, non_blocking=True)
        input_times = batch["time"].reshape(batch_size, seq_len, 1).cuda(cuda_number, non_blocking=True)
        target_times = batch["time_target"].cuda(cuda_number, non_blocking=True)
        
        output_locations, output_times = temp_generator([input_locations, input_times], labels)
        output_locations_v = output_locations.view(-1,output_locations.shape[-1])
        output_times_v = output_times.view(-1,output_times.shape[-1])
        # print(output_v.shape)

        loss_location = loss_model(output_locations_v, target_locations.view(-1))
        loss_time = loss_time_fn(output_times_v, target_times, ignore_value=ignore_index)
        loss = loss_location + loss_time

        grad = torch.autograd.grad(loss, temp_generator.parameters())
        # add noise following the gausiann distribution with the scale of noise_scale 
        fast_weights = dict(map(lambda p: (p[1][0],p[1][1] - lr * p[0] + noise_scale*torch.randn(p[0].data.size()).cuda(cuda_number)), zip(grad, temp_generator.named_parameters())))
        output_locations, output_times = temp_generator([input_locations, input_times], labels, fast_weights)

        # conver input_times to indice of time_ranges
        distributions = [[] for _ in time_ranges]
        for i in range(batch_size):
            for j in range(seq_len):
                if input_times[i,j].item() == ignore_index:
                    continue
                index = time_to_index(input_times[i,j].item(), time_ranges)
                # print(index)
                distributions[index].append(torch.exp(output_locations[i,j,:]))

        loss = 0
        # get the mean of the distributions
        for i in range(len(distributions)):
            if len(distributions[i]) == 0:
                continue
            distributions[i] = torch.stack(distributions[i]).mean(dim=0)
            loss += loss_kl(distributions[i], global_distributions[i])/len(distributions)
            # print(i, distributions[i], global_distributions[i])


        # output_locations_probs = torch.exp(output_locations)[:,:,:-1]
        # loss = loss_kl(output_locations_probs, input_times, global_distributions)
        # print(output_locations_probs)
        # print(loss.item())
        loss.backward()
        meta_optim.step()
        meta_optim.zero_grad()
    # copy the parameters of temp_generator to generator._module
    generator._module.load_state_dict(temp_generator.state_dict())

    return generator.cuda(cuda_number)


def pre_training(data_loader, temp_generator, generator, lr, global_distributions, time_ranges, loss_kl, ignore_index, n_iter):
    cuda_number = next(generator.parameters()).device.index

    # deepcopy the parameters of generator._module to temp_generator
    temp_generator.load_state_dict(generator._module.state_dict())
    # move the temp_generator to cuda
    temp_generator = temp_generator.cuda(cuda_number)

    optimizer = optim.Adam(temp_generator.parameters(), lr=lr)

    count = 0

    while True:
            
        for batch in data_loader:
            
            # break the while if count >= n_iter
            if count >= n_iter:
                break

            count += 1
                
            input_locations = batch["input"].cuda(cuda_number, non_blocking=True)

            batch_size = input_locations.shape[0]
            seq_len = input_locations.shape[1]

            labels = batch["label"].cuda(cuda_number, non_blocking=True)
            input_times = batch["time"].reshape(batch_size, seq_len, 1).cuda(cuda_number, non_blocking=True)
            
            output_locations, _ = temp_generator([input_locations, input_times], labels)

            # conver input_times to indice of time_ranges
            distributions = [[] for _ in time_ranges]
            for i in range(batch_size):
                for j in range(seq_len):
                    if input_times[i,j].item() == ignore_index:
                        continue
                    index = time_to_index(input_times[i,j].item(), time_ranges)
                    # print(index)
                    distributions[index].append(torch.exp(output_locations[i,j,:]))

            loss = 0
            # get the mean of the distributions
            for i in range(len(distributions)):
                if len(distributions[i]) == 0:
                    continue
                distributions[i] = torch.stack(distributions[i]).mean(dim=0)
                loss += loss_kl(distributions[i], global_distributions[i])/len(distributions)

            loss.backward()
            optimizer.step()
        else:
            continue
        break
        


    # copy the parameters of temp_generator to generator._module
    generator._module.load_state_dict(temp_generator.state_dict())

    return generator