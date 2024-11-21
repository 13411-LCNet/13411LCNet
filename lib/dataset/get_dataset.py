import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.vocdataset import VOCdataset
from dataset.ropedataset import RopeDataset
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )   

        test_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )   

    elif args.dataname == 'rope' or args.dataname == 'fiberrope':
        dataset_dir = args.dataset_dir

        train_dataset = RopeDataset(
            osp.join(dataset_dir, 'ropeTrain'),
            osp.join(dataset_dir, 'imageLabelsTrain.csv'),
            train_data_transform
        )

        val_dataset = RopeDataset(
            osp.join(dataset_dir, 'ropeVal'),
            osp.join(dataset_dir, 'imageLabelsVal.csv'),
            test_data_transform
        )


        test_dataset = RopeDataset(
            osp.join(dataset_dir, 'ropeTest'),
            osp.join(dataset_dir, 'imageLabelsTest.csv'),
            train_data_transform
        )

    elif args.dataname == 'voc' or args.dataname == 'voc2007':

        dataset_dir = args.dataset_dir
        train_dataset = VOCdataset(
            rootDir= osp.join(dataset_dir, 'train'),
            imgType= "train",
            anno_path=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC/pascal_train2007.json'),
            input_transform=train_data_transform,
        )
        val_dataset = VOCdataset(
            rootDir= osp.join(dataset_dir, 'val'),
            imgType= "val",
            anno_path=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC/pascal_val2007.json'),
            input_transform=test_data_transform,  
        )   

        test_dataset = VOCdataset(
            rootDir= osp.join(dataset_dir, 'test'),
            imgType= "test",
            anno_path=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC/pascal_test2007.json'),
            input_transform=test_data_transform,  
        )  

        # train_dataset = VOCdataset(
        #     rootDir=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC'),
        #     image_dir=osp.join(dataset_dir, 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'),
        #     anno_path=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC/pascal_train2007.json'),
        #     input_transform=train_data_transform,
        # )
        # val_dataset = VOCdataset(
        #     rootDir=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC'),
        #     image_dir=osp.join(dataset_dir, 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'),
        #     anno_path=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC/pascal_val2007.json'),
        #     input_transform=test_data_transform,  
        # )   

        # test_dataset = VOCdataset(
        #     rootDir=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC'),
        #     image_dir=osp.join(dataset_dir, 'VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'),
        #     anno_path=osp.join(dataset_dir, 'PASCAL_VOC/PASCAL_VOC/pascal_test2007.json'),
        #     input_transform=test_data_transform,  
        # ) 


    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    # print("Train dataset type: ", type(train_dataset))
    # print("Val dataset type: ", type(val_dataset))
    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))
    return train_dataset, val_dataset, test_dataset
