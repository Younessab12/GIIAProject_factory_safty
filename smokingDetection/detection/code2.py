import code
c = int(input("enter cam input: "))
detector=code.Detector("best.pt",c)
detector.detect()
