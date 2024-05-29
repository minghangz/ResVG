# ResVG

This repository is the official Pytorch implementation for paper **ResVG: Enhancing Relation and Semantic Understanding in Multiple Instances for Visual Grounding**.

## Contents

1. [Usage](#usage)
2. [Results](#results)
3. [Contacts](#contacts)
4. [Acknowledgments](#acknowledgments)

## Usage

### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- Check [requirements.txt](requirements.txt) for other dependencies. 

### Data Preparation
You can download the images from the original source and place them in `./data/image_data` folder:
- RefCOCO
```
/network_space/storage43/ln_data/images/train2014
```
-ReferItGame
```
/network_space/storage43/ln_data/referit/ReferIt
```
-Flickr30K Entities
```
/network_space/storage43/ln_data/flickr/flickr
```

The training samples can be download from [data](). Finally, the `./data/` and `./image_data/` folder will have the following structure:

```

|-- data
      |-- flickr
      |-- gref
      |-- gref_umd
      |-- referit
      |-- unc
      |-- unc+
```

### Pretrained Checkpoints
1.You can download the DETR checkpoints from [detr_checkpoints](https://disk.pku.edu.cn:443/link/4E6B5343270CC07E52A88AA8A7A31CE8). These checkpoints should be downloaded and move to the checkpoints directory.

```
mkdir checkpoints
mv detr_checkpoints.tar.gz ./checkpoints/
tar -zxvf detr_checkpoints.tar.gz
```

### Training and Evaluation

1.  Training on RefCOCO. 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --config configs/ResVG_R50_unc+.py
    ```

2.  Evaluation on RefCOCO.
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --use_env test.py --config configs/ResVG_R50_unc.py --checkpoint ResVG_R50_unc.pth --batch_size_test 32 --test_split val;
    ```

## Results

<table border="2">
    <thead>
        <tr>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCO </th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCO+</th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCOg</th>
            <th colspan=1> ReferItGame</th>
            <th colspan=1> Flickr30K</th>
        </tr>
    </thead>
    <tbody>
    <tr>    
            <td>val</td>
            <td>testA</td>
            <td>testB</td>
            <td>val</td>
            <td>testA</td>
            <td>testB</td>
            <td>g-val</td>
            <td>u-val</td>
            <td>u-test</td>
            <td>test</td>
            <td>test</td>
        </tr>
    </tbody>
    <tbody>
    <tr>
            <td>85.51</td>
            <td>88.76</td>
            <td>79.93</td>
            <td>73.95</td>
            <td>79.53</td>
            <td>64.88</td>
            <td>73.13</td>
            <td>75.77</td>
            <td>74.53</td>
            <td>72.35</td>
            <td>79.52</td>
        </tr>
    </tbody>
</table>
