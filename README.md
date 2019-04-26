# prac_tensorflow

![Alt text](cat01.jpg?raw=true "Title")

## Setup
  ### Prerequisites
   - Python version 3.6.5
   https://www.python.org/downloads/release/python-365/
   
### Installation

To install all dependencies run the following command:

```
python -m pip install -r requirements.txt

When you already installed
  try
    import tensorflow
    
if "import tensorflow" has a problem like this : DDL Error
download this one to fix
  https://www.microsoft.com/en-us/download/confirmation.aspx?id=53587
```


## Download dataset from this link
  https://drive.google.com/open?id=1dExFDSGBDHbZlAxnzDWo44lWnWKS5Kc1
  
## Extract dataset
  Given the following directory structure:
  ```
    images:
    |---cat
    |     |---persia.png
    |     |---...
    |---dog
    |     |---dog1.png
    |     |---...
   ```
      
## train the model
  python train.py

## predic
  python predic.py
