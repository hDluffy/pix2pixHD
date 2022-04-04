import torch.utils.data
from data.base_data_loader import BaseDataLoader
import torch.distributed as dist


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.use_ddp=opt.use_ddp
        if self.use_ddp:
            word_size = dist.get_world_size()
            self.train_sampler=torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas = word_size,
                rank = opt.rank
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=int(opt.nThreads),
                pin_memory=True,
                sampler=self.train_sampler)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))

    def load_data(self):
        if self.use_ddp:
            return self.dataloader,self.train_sampler
        else:
            return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
