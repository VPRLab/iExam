# iExam

$ git clone https://YourUserName@github.com/VPRLab/ProjectName

// make you local change

$ git add .

$ git status //check the files you want to commit

$ git commit -a

$ git push origin main

# Install on Mac
- % sudo python3 -m pip install PyQt5
- % sudo python3 -m pip install opencv-python
- % sudo python3 -m pip install pytesseract
- % sudo python3 -m pip install torch
- % sudo python3 -m pip install torchvision
- % sudo python3 -m pip install matplotlib
- Install the tesseract binary via brew and make sure the command of "tesseract --help" can run.
  See https://brew.sh/, https://guides.library.illinois.edu/c.php?g=347520&p=4121425 and https://tesseract-ocr.github.io/tessdoc/Installation.html
- Still need to explicitly install the tesseract language package: % brew install tesseract-lang (installing tesseract-eng failed).
