# face-detection

train from data/label_name/*

    ./face_detector.out -train

save images to to data/label_name/*


    ./face_detector.out -camera --label my_name


update model based on the new labels

    ./face_detector.out -update

predict from faces

    ./face_detector.out -predict




Example Datasets For Training:

    https://data.caltech.edu/records/6rjah-hdv18

    https://vis-www.cs.umass.edu/lfw/#resourcesa