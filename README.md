# Github Issue Title Generator

Uses google brain's [Tensor2Tensor library](https://github.com/tensorflow/tensor2tensor) to recreate work from [this article](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)

Currently WIP, as I haven't been able to reproduce work from the original article.

## Results

### [Input Issue](https://github.com/tensorflow/tensor2tensor/issues/518)

I use 1.4.1 version of T2T with 2 * 1080ti and BS of 2048. When I trained translate_ende_wmt32k with big_single_gpu model for ~ 600k steps, I got a BLEU score of 26.02, which is only ~2 BLEUs less than reported in the "Attention is all you need" paper. But when I use the same set-up and parameters to train translate_enfr_wmt32k model - I get around 33 BLEU after 1.4mln steps, which is whole 9 BLEU points less than result in the paper. Which seems somewhat too much to be compensated with number of GPUs. In gitter @lukaszkaiser assumed that it might be a tokenization issue. I'm not sure what's the issue at this point, but results look somewhat suspicious to me. Hope you could advise what might be happening here.

### Predicted Title

use 1.4.1 version of 22 with 2 wi  1080ti and   of 2048. hen n  trained translate_ende_wmt32k with big_single_gpu

### [Input Issue](https://github.com/tensorflow/tensorflow/issues/19126)

I have successfully built Tensorflow C++ Windows with MSVC2015, but currently I am interested in building Tensorflow C++ Windows with MSVC2013. May I know has anybody done this before, or does it doable?

### Predicted Title

have successfully built eensorflow +++ indows with , inde55, but currently but am interested in building ensorflow +++ inndows with done 013. aay d know has anybody done this before, or does it doable?
