# To run the project
1. clone the repo
2. make sure to have python3 and pip commands on your local machine terminal
3. run **pip install --user pipenv**
4. run **pipenv install**
5. run **pipenv shell** to spawn a shell in the virtual env of the project; exit:ctrl+d
6. run **python3 [Any].py**

## NOTES
https://kezunlin.me/post/61d55ab4/  
check available options here to loop over numpy arrays efficiently  
ALSO beware that Numba is installed in the pipfile

To optimize circles drawing, draw each modulated circle first, then copy them instead of 
drawing a new one every time

## Additional notes
BSM is the most favorable one due to the following reasons:  
  1. BCM and BSM are in general faster to compute than ACM and ASM  
  2. BSM shows a more realistic effect to phosphene in comparison to BCM  
  3. Applying BSM to the same image many times accumulated does not change the phosphene effect  
  4. Applying BCM to the same image many times accumulated, however, yields a white image

3 & 4 describe the code below:
```
while time.time() - start < 1:
        counter += 1
        img = pSim(img, simode = Simode.BCM) 
        # notice that img is repeatedly used as an input and an output to the simulation
```
