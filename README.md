# LLM-Op-PerfTest

## Conda Environment Setup

Please Use or refer to the scripts 'create-conda-env.sh'

```
	bash create-conda-env.sh
```

TO run the code of this dictory, please add the $PWD to $PYTHONPATH.

```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 🚀 How to run

1️⃣  Firstly, you need to use scripts/sim-gamma.py to make the sample-total-tokens-{N}.out

Or, you can use the files which in the dictory of  'sample-out/'.

---

2️⃣ Then, you can perf the operation of flash-attn using the sample.out

```
	bash test.sh
```


