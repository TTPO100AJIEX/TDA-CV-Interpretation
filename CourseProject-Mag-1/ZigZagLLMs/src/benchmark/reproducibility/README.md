## Reproducibility
In this folder the objective is to get the best starting layer to a given block of number of layers by using the algorithms from [Angular distance](https://arxiv.org/abs/2403.17887v1) and [Bi-Score](https://arxiv.org/abs/2403.03853). 

The code used is the one of [short-transformers](https://github.com/melisa-writer/short-transformers) with slightly modification to get the resuls from Pythia 6.9B.

Note that the official library of [short-transformers](https://github.com/melisa-writer/short-transformers) downloadable via ```pip install short-transformers``` is used for all the modeles except pythia, because for pythia it will be used the modified version inside the folder ***pythia-version***.


The ```--method``` parameter can only be ```"biscore"``` or ```"angular"```.

```bash
python starting_layer.py --model="meta-llama/Llama-2-7b-hf"\
                         --block_number=5\
                         --token="your Hugging Face token"\
                         --method="biscore"
```

### Model names used in this work

 | Model  | ```--model``` input |
 |:------:|:---------:|
 |Llama 2 7B | meta-llama/Llama-2-7b-hf  |
 |Llama 2 13B | meta-llama/Llama-2-13b-hf |
 |Llama 2 70B | meta-llama/Llama-2-70b-hf |
 |Llama 3 8B  | meta-llama/Meta-Llama-3-8B |
 |Llama 3 70B | meta-llama/Meta-Llama-3-70B |
 |Pythia 6.9B | EleutherAI/pythia-6.9b-deduped |
 |Mistral 7B  | mistralai/Mistral-7B-v0.1|

```
short-transformers
Repository: https://github.com/melisa-writer/short-transformers

License: MIT License

Copyright: 2023 Melisa

This project includes code from the short-transformers repository, which is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
Wikipedia

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
Wikipedia

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```