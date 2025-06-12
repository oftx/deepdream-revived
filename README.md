# deepdream-revived
A version of DeepDream that works in (relatively) modern-day python.

The original [DeepDream](https://github.com/google/deepdream) and related projects like [DeepDreamVideo](https://github.com/graphific/DeepDreamVideo) were written 10 years ago in python 2, so they're effectively dead for anyone interested in running it today. I've attempted to port this functionality to run in the most recent version of python 3 possible.

Note that I'm not an expert with this stuff. It's simply my best effort to preserve old tech.

## Installation

***WARNING***: This setup is not noob friendly. You need some familiarity with python virtual environments and a willingness to troubleshoot platform-specific depencency issues. Please  read carefully as there are some important caveats to know before you try to run it.

The most recent version of python that works for me is **python 3.10**. The main dependency that hinders newer python versions is [tensorflow](https://www.tensorflow.org/install). Supposedly tensorflow works for python 3.8-3.11, but I personally didn't have success with 3.11.

To get python 3.10, you may have to compile it from source, find a legacy installer, or find an unofficial package, such as the [deadsnakes](https://linuxcapable.com/how-to-install-python-3-10-on-ubuntu-linux/) repository for Ubuntu or [python310](https://aur.archlinux.org/packages/python310) from the Arch Linux AUR.


- If you're on **Windows+Nvidia**, note that you'll need to install the [Microsoft Visual C++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- If you're on **Windows+AMD**, you may be able to get it to work through [WSL](https://wiki.archlinux.org/title/Install_Arch_Linux_on_WSL)
- If you're on **Linux+AMD** (like me), you'll have to install [tensorflow-rocm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html)
- If you're on **Linux+Nvidia** it may work as-is, but you might have to select a specific version of tensorflow (see instructions below)

You may be asking, why doesn't this use pytorch? My implementation is based on [this tutorial](https://www.tensorflow.org/tutorials/generative/deepdream) from tensorflow. DeepDream is very old and predates pytorch so that's why this implementation is so wonky. I don't know enough about pytorch or tensorflow to know how to port it. I welcome anyone who has ideas on how to do that, because eliminating the tensorflow dependency would make life much easier.

### Installation Steps
1. Setup a virtual environment:
```
python3.10 -m venv venv
```
2. Install tensorflow

For Nvidia:
```
pip install tensorflow
```
For AMD:
```
pip install tensorflow-rocm
```
However, the tensorflow-rocm in pip is very out of date. So you may want to install a newer release:
```
pip install --upgrade tensorflow-rocm==2.16.2 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/
```
I couldn't get anything newer to work, but feel free to try other releases by browsing [here](https://repo.radeon.com/rocm/manylinux/).\
Despite my best efforts, I still can't get GPU inference to work, so I need `tensorflow-rocm` **and** I need to specify `--cpu` when running inference. Good luck and if you get GPU inference to work on AMD, let me know what specific configuration works for you.

***NOTE***: The specific version of tensorflow that you install, and usage of CPU vs GPU, can drastically change what the output looks like. I don't know why this is the case. If you want to get results very close to the [tutorial](https://www.tensorflow.org/tutorials/generative/deepdream), use the pip version [tensorflow-rocm 2.14.0.600](https://pypi.org/project/tensorflow-rocm/2.14.0.600/) from January 2024 or [tensorflow 2.14.0](https://pypi.org/project/tensorflow/2.14.0/) from September 2023.

3. Install additional requirements:
```
pip install -r requirements.txt
```

## Running Inference
By default, the script is configured to run a simple example that loads `example.png` from the current working directory, so you can test your setup by simply running:
```
python dream.py
```
Outputs are put into the `output` directory.

If you're having problems with GPU inference, try CPU only:
```
python dream.py --cpu
```
Specify your own image with `-i/--input`:
```
python dream.py -i input.jpg
```

### Command Line Flags
- `--cpu` Disables GPU inference, running in CPU only mode
- `--steps` Sets the number of inference steps (default is 100)
  - In my case, using `tensorflow-rocm==2.16.2` produces similar outputs with drastically fewer steps than `2.14.0.600`, like 4-10 steps instead of 100.  
- `--step_size` Sets the size of each step (default is 0.1)
- `--mode` Sets the inference mode, which can be `"simple"` or `"octaves"`. Simple mode takes the least amount of time, octaves mode is more flexible and can produce different patterns depending on how you specify the octaves.