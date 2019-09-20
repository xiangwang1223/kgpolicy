from torch.utils.data import DataLoader
from .data_generator import Train_Generator, Test_Generator, Sampler_Generator

def build_loader(args_config):
    train_generator = Train_Generator(args_config=args_config)
    train_loader = DataLoader(train_generator, batch_size=args_config.batch_size, shuffle=True,
                              num_workers=args_config.num_threads)

    test_generator = Test_Generator(args_config=args_config)
    test_loader = DataLoader(test_generator, batch_size=args_config.test_batch_size, shuffle=False,
                             num_workers=args_config.num_threads)

    sampler_generator = Sampler_Generator(args_config=args_config)
    sampler_loader = DataLoader(sampler_generator, batch_size=args_config.batch_size, shuffle=True,
                            num_workers=args_config.num_threads)
    return train_loader, test_loader, sampler_loader
