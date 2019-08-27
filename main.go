package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"os"
	"strconv"
)
import "github.com/sjwhitworth/golearn/knn"
import "github.com/sjwhitworth/golearn/base"
import "github.com/sjwhitworth/golearn/evaluation"

const (
	datafile = "data/magictelescope_csv.csv"
)

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
		fmt.Println(err)
	}
	r := bufio.NewReader(f)
	// parse data into the data frame type.
	df := dataframe.ReadCSV(r)
	fmt.Println(df)
	fmt.Println(df.Describe())

	fmt.Println(df.Select([]string{"ID", "fWidth:", "fSize:", "class:"}))

	if err != nil {
		fmt.Println("os.Open: ", err)
	}
	// read in file as csv
	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("ReadAll: ", err)
	}
	size := len(records)
	fmt.Println("size: ", size)
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
		fmt.Println(err)
	}
	// make plot formatter
	p, err := plot.New()
	if err != nil {
		fmt.Println(err)
	}
	// label plot
	p.Title.Text = "Width vs Size"
	p.X.Label.Text = "Width"
	p.Y.Label.Text = "Size"
	p.Add(scatter)
	w, err := p.WriterTo(8*vg.Inch, 8*vg.Inch, "png")
	if err != nil {
		panic(err)
	}
	// display inside notebook
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	build_cnn()
}

func build_cnn() {
	dataCSV, err := base.ParseCSVToInstances(datafile, true)
	if err != nil {
		fmt.Println(err)
	}

	k := knn.NewKnnClassifier("euclidean", "kdtree", 3)

	// Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(dataCSV, 0.3)
	x, y := trainData.Size()
	w, z := testData.Size()
	fmt.Println(x, y, w, z)
	k.Fit(trainData)
	// Calculates the Euclidean distance and returns the most popular label
	predictions, err := k.Predict(testData)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		fmt.Println("Unable to get confusion matrix: %s", err.Error())
	}
	fmt.Println(confusionMat)
	accu := evaluation.GetAccuracy(confusionMat)
	fmt.Println(accu)
}
