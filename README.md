# LLPackNet
The project is the official implementation of our BMVC 2020 paper, "Towards Fast and Light-Weight Restoration of Dark Images"

We show that we can enhance High Resolution,2848×4256, extremely dark single-image in the ballpark of 3 seconds even on a CPU. We achieve this with 2−7× fewer model parameters, 2−3× lower memory utilization, 5−20× speed up and yet maintain a competitive image reconstruction quality compared to the state-of-the-art algorithms. 

Watch the below video for results and overview of LLPackNet.

[![Watch the project video](https://raw.githubusercontent.com/MohitLamba94/LLPackNet/master/pics/video.png)](https://www.youtube.com/watch?v=nO6pizVH_qM&feature=youtu.be)

# How to use the code?
The ```train.py``` and ```test.py``` files were used for training and testing. Relevant comments have been added to these files for better understanding. You however need to download the [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark) in your PC to execute them. 

The Jupyter Notebooks containing test code for the ablation studies can be also found in the ```ablations``` folder.

We used PyTorch version 1.3.1 with Python 3.7 to conduct the experiment. Along with the commonly used Python libraries such Numpy and Skimage, do install the [Rawpy](https://pypi.org/project/rawpy/) library required to read RAW images.



# Cite us
If you find any information provided here useful please cite us,

```
@inproceedings{lamba2020LLPackNet,
  title={Towards Fast and Light-Weight Restoration of Dark Images},
  author={Lamba, Mohit and Balaji, Atul and Mitra, Kaushik},
  booktitle={British Machine Vision Conference (BMVC) 2020},
  year={2020},
  organization={BMVC}
}
```
