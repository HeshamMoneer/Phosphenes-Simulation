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
This branch is dedicated to displaying the different, 16, levels of the available modulated phosphenes
