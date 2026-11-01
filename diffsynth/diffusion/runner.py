import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _collate_single_item(batch):
    return batch[0]


def _build_dataloader(dataset, shuffle, num_workers):
    kwargs = {
        "dataset": dataset,
        "shuffle": shuffle,
        "collate_fn": _collate_single_item,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        # Petrel's C++ S3 runtime is process-global. Using spawn prevents
        # DataLoader workers from inheriting an initialized runtime via fork.
        kwargs.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
            prefetch_factor=4,
        )
    return torch.utils.data.DataLoader(**kwargs)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = _build_dataloader(dataset, shuffle=True, num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # if dataset.load_from_cache:
                #     loss = model({}, inputs=data)
                # else:
                model_output = model(data)
                if isinstance(model_output, tuple):
                    loss, global_valid_count = model_output
                else:
                    loss = model_output
                    global_valid_count = None
                accelerator.backward(loss)
                if global_valid_count == 0:
                    # Backward is still required to finish DDP reductions, but
                    # a globally rejected batch must not apply AdamW decay or
                    # advance checkpoint/scheduler steps.
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = _build_dataloader(dataset, shuffle=False, num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
