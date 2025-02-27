---
author: "Jorge Roldan"
date: '2025-02-22'
title: 'Setting up a Conda environment'
tags: ['python', 'conda']
categories: ['tutorials']
ShowToc: true
ShowBreadCrumbs: false
---

# TLDR

```bash
conda create --name my_conda_env python=3.11
```

```bash
conda env list
```

```bash
conda activate my_conda_env
```

```bash
pip install <package_name>
```
or 

```bash
pip install -r requirements.txt 
```

# What's a conda environment?
Knowing how to set up a Conda environment is an essential skill for any data scientist or Python developer.  Conda is
an open source package management system  for Python.  

In this post, I will show you how to set up a Conda environment for your project, doing this, will help you to easily install and use 
any dependency you will need. 


# Create conda environment
## Install and validate your anaconda or conda installation 
For this tutorial, you will need to install [Python](https://www.python.org/downloads/) and [Anaconda](https://www.anaconda.com/download/), I recommend downloading miniconda, which is a lighter version of Anaconda.

Before creating a conda environment, you need to check if conda is installed in your system. To do this, open your terminal and type:

```bash
conda --version
```

You should see something like this:

```bash
conda 4.12.0  
````

Now, make sure you have the latest version of conda by running:

```bash
conda update conda
```

## Create the environment
You are now ready to create your first conda environment. To do this, run:

```bash
conda create --name my_conda_env python=3.10
```

To specify the path where you want to create the environment, use the `--prefix` or `-p` flag. For example, to create the environment in the `default_path` directory, run: 

```bash
conda create --prefix <default_path>/my_conda_env python=3.10
````

Make you created the environment by running the following command:

```bash
conda env list
```
```bash
# conda environments:
#
base                  *  /Users/Jroldan001/opt/anaconda3
my_conda_env             /Users/Jroldan001/opt/anaconda3/envs/my_conda_env
```

As you can see, the environment you just created is listed in the output, but it is not active yet. To activate it, run:

```bash
conda activate /Users/Jroldan001/opt/anaconda3/envs/my_conda_env
```
Now the `*` symbol is next to the environment you just activated.

```bash
# conda environments:
#
base                     /Users/Jroldan001/opt/anaconda3
my_conda_env          *  /Users/Jroldan001/opt/anaconda3/envs/my_conda_env
```

## Check `python` and `pip`
Now that you created your environment, make sure that you are using the correct `python` and `pip`. For this run 
`which python` and `which pip`. You should see the same path as the one you used to create the environment.

# Install packages using pip
Now that you have your environment set up, you can install any package you need. For this, you can use `pip`. For example, to install `numpy` run:

```bash 
pip install numpy
``` 
You can check that the package was installed by running:

```bash 
pip freeze
```

And you will see the following
```
numpy==1.25.2
```

## Install packages from a requirements.txt file
If you have a `requirements.txt` file, you can install all the packages listed in it by running:


```bash
pip install -r requirements.txt 
```
You can check that all packages were installed by again running the `pip freeze` command.

# Deactivate the environment
To deactivate the environment, run:

```bash 
conda deactivate
```

As you will see, the `*` symbol is now next to the `base` environment.

```bash
# conda environments:
#
base                  *  /Users/Jroldan001/opt/anaconda3
my_conda_env            /Users/Jroldan001/opt/anaconda3/envs/my_conda_env
```