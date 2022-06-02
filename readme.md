# To run the project
1. clone the repo
2. make sure to have python3 and pip commands on your local machine terminal
3. run **pip install --user pipenv**
4. run **pipenv install**
5. run **pipenv shell** to spawn a shell in the virtual env of the project; exit:ctrl+d
6. run **python3 [Any].py**  
ANOTHER APPROACH  
7. instead of spwaning a pipenv shell, you can run **pipenv run** + some script from the *pipfile*

## NOTES
https://kezunlin.me/post/61d55ab4/  
check available options here to loop over numpy arrays efficiently  
ALSO beware that Numba is installed in the pipfile

To optimize circles drawing, draw each modulated circle first, then copy them instead of 
drawing a new one every time

## Additional notes
Contrast brightness should always come before modulation
The performance after optimization is ~300 frames per second
The performance can be further optimized by passing the circles cache between frames

## All optimizations that I worked on
1. Used Gaussian blur kernel instead of pdf()  
2. The faces are not detected in every frame but rather limited times in 1s (3 times/second)  
3. switched to optimized numpy loops  
4. cached circles to avoid calling the draw method of opencv many times  
5. move gaussian effect to be done on phosphenes before caching did not show performance improvement
6. saving cache between frames raised the throughput


## SFR-ROI magnification
1. The nose position could be directly detected using landmark detection point @33

## Phosphene simulation throughput optimization
The throuput was improved from 170 fps to ~300 fps initially
After caching the blur effects as well, the throuput jumped to ~500 fps
