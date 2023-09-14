# Statistical analysis
Python script for statistical analysis. 

To find variables that differentiate 10 groups. 

In my example, this are 10 types of serotonin receptors.

## Instruction

1. Create conda environment

To create conda environment use this code in the console:

```bash
$ conda env create -f environment.yml
```

2. Activate conda environment
```bash
$ conda activate for_statistical_analysis
```

3. Change the upper part of the script

For your own dataset change 'file', 'grouping_variable' (column which consists of information about groups),
'file_name' (core part of future files with results) and assumption of alpha value. In my research, it was 0.05. 


4. Use the script
```bash
$ python statistical_analysis.py
```

