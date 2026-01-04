import argparse
argsall = argparse.Namespace(
    testdata_path='data/mr2ct/test/mr',
    ckpt = 'pretrained_model/ct_ddpm.pth',
    dsepath = 'pretrained_model/mr2ct_dse.pt',
    config_path = 'profiles/mr2ct/mr2ct.yml',
    t = 500,
    ls =  500.0,
    li = 2.0,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step= 1,
    batch_size = 20,
    diffusionmodel = 'DDPM',
    down_N = 32,
    seed=1234)
