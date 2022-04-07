from scipy.stats import norm

def gaussArr(size, lastValue = 4):
  array = []
  step = lastValue / size
  peak = norm.pdf(0)
  for i in range(size):
    x = i * step
    array.append(norm.pdf(x)/peak)
  return array

def main():
  print(str(gaussArr(10)))

if __name__ == "__main__":
    main()