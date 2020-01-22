from torch.utils.data import DataLoader
from common.dataset.dataset import TrainGenerator, TestGenerator


def build_loader(args_config, graph):
    train_generator = TrainGenerator(args_config=args_config, graph=graph)
    train_loader = DataLoader(
        train_generator,
        batch_size=args_config.batch_size,
        shuffle=True,
        num_workers=args_config.num_threads,
    )

    test_generator = TestGenerator(args_config=args_config, graph=graph)
    test_loader = DataLoader(
        test_generator,
        batch_size=args_config.test_batch_size,
        shuffle=False,
        num_workers=args_config.num_threads,
    )

    return train_loader, test_loader
