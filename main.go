package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/kniren/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	datafile = "data/magictelescope_csv.csv"
)

var (
	WarningLogger *log.Logger
	InfoLogger    *log.Logger
	ErrorLogger   *log.Logger
)

func init() {
	file, err := os.OpenFile("log.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalln("Failed to open log file", err)
	}

	InfoLogger = log.New(os.Stdout, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
	multi := io.MultiWriter(file, os.Stdout)
	WarningLogger = log.New(multi, "WARNING: ", log.Ldate|log.Ltime|log.Lshortfile)
	ErrorLogger = log.New(multi, "ERROR: ", log.Ldate|log.Ltime|log.Lshortfile)
}

type GammaImage struct {
	ID       int
	FLength  float64
	FWidth   float64
	FSize    float64
	FConc    float64
	FConcl   float64
	FAsym    float64
	FM3Long  float64
	FM3Trans float64
	FAlpha   float64
	FDist    float64
	Class    string
}

func main() {
	f, err := os.Open(datafile)
	if err != nil {
		ErrorLogger.Fatalln("Failed to open data file")
	}
	r := bufio.NewReader(f)
	// parse data into the data frame type.
	df := dataframe.ReadCSV(r)
	InfoLogger.Println(df)
	InfoLogger.Println(df.Describe())

	InfoLogger.Println(df.Select([]string{"ID", "fWidth:", "fSize:", "class:"}))

	// read in file as csv
	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		ErrorLogger.Println("ReadAll: ", err)
	}
	size := len(records)
	InfoLogger.Println("size: ", size)
	images := make([]GammaImage, size)
	//store data for making plot
	for idx, img := range records {
		// fmt.Println(img)
		// fmt.Printf("%T, %v\n", img, img)
		if idx != 0 {
			ID, _ := strconv.Atoi(img[0])
			FLength, _ := strconv.ParseFloat(img[1], 64)
			FWidth, _ := strconv.ParseFloat(img[2], 64)
			FSize, _ := strconv.ParseFloat(img[3], 64)
			FConc, _ := strconv.ParseFloat(img[4], 64)
			FConcl, _ := strconv.ParseFloat(img[5], 64)
			FAsym, _ := strconv.ParseFloat(img[6], 64)
			FM3Long, _ := strconv.ParseFloat(img[7], 64)
			FM3Trans, _ := strconv.ParseFloat(img[8], 64)
			FAlpha, _ := strconv.ParseFloat(img[9], 64)
			FDist, _ := strconv.ParseFloat(img[10], 64)
			image := GammaImage{
				ID:       ID,
				FLength:  FLength,
				FWidth:   FWidth,
				FSize:    FSize,
				FConc:    FConc,
				FConcl:   FConcl,
				FAsym:    FAsym,
				FM3Long:  FM3Long,
				FM3Trans: FM3Trans,
				FAlpha:   FAlpha,
				FDist:    FDist,
				Class:    img[11],
			}
			images = append(images, image)
		}
	}
	f.Close()
	pts := make(plotter.XYs, len(images))
	for i, img := range images {
		pts[i].X = img.FWidth
		pts[i].Y = img.FSize
	}
	// make new scatter plot
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		WarningLogger.Println("NewScatter: ", err)
	}
	// make plot formatter
	p, err := plot.New()
	if err != nil {
		WarningLogger.Println(err)
	}
	// label plot
	p.Title.Text = "Width vs Size"
	p.X.Label.Text = "Width"
	p.Y.Label.Text = "Size"
	p.Add(scatter)
	w, err := p.WriterTo(8*vg.Inch, 8*vg.Inch, "png")
	if err != nil {
		ErrorLogger.Fatalln(err)
	}

	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	build_cnn()
}

func build_cnn() {
	dataCSV, err := base.ParseCSVToInstances(datafile, true)
	if err != nil {
		WarningLogger.Println(err)
	}

	k := knn.NewKnnClassifier("euclidean", "kdtree", 3)

	// Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(dataCSV, 0.3)
	x, y := trainData.Size()
	w, z := testData.Size()
	InfoLogger.Println(x, y, w, z)
	k.Fit(trainData)
	// Calculates the Euclidean distance and returns the most popular label
	predictions, err := k.Predict(testData)
	if err != nil {
		WarningLogger.Println(err)
	}
	InfoLogger.Println("Predictions: ", predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		WarningLogger.Printf("Unable to get confusion matrix: %s", err.Error())
	}
	InfoLogger.Println(confusionMat)
	accu := evaluation.GetAccuracy(confusionMat)
	InfoLogger.Println(accu)
}
