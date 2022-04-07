# Robust Peak Detection for Holter ECGs by Self-Organized Operational Neural Networks


This repository includes the implentation of R peak detection method in [Robust Peak Detection for Holter ECGs by Self-Organized Operational Neural Networks](https://ieeexplore.ieee.org/abstract/document/9743556).

## Network Architecture
![image](https://user-images.githubusercontent.com/43520052/162131931-0856cc7f-4065-43d4-b1ec-2d664b4ffd3f.png)

## Dataset

- [The China Physiological Signal Challenge 2020](http://2020.icbeb.org/CSPC2020), (CPSC-2020) dataset is used for training & testing.
- R peak annotations are already available in the data folder.


## Run

#### Train
- Download CPSC data from the link to the "data/" folder
- Data Preparation without augmentation
```http
  python prepare_data.py
```
- Data Preparation with augmentation
```http
  python prepare_data_augmentation.py
```
- Start patient wise training and evaluation.
```http
  python run_selfONN.py
```



## Citation

If you use the provided method in this repository, please cite the following paper:

```
@article{gabbouj2022robust,
  title={Robust Peak Detection for Holter ECGs by Self-Organized Operational Neural Networks},
  author={Gabbouj, Moncef and Kiranyaz, Serkan and Malik, Junaid and Zahid, Muhammad Uzair and Ince, Turker and Chowdhury, Muhammad EH and Khandakar, Amith and Tahir, Anas},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```
