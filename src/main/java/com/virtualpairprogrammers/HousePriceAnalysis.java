package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceAnalysis
{
    public static void main(String[] args)
    {
        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark =  SparkSession.builder()
                .appName("House Price Analysis").master("local[*]")
                .config("spark.sql.warehouse.dir","file:///c:/tmp/")
                .getOrCreate();


        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/kc_house_data.csv");

//        csvData.printSchema();
//        csvData.show();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] {"bedrooms","bathrooms","sqft_living"})
                .setOutputCol("features");

        Dataset<Row> modelInputData = vectorAssembler.transform(csvData)
                .select("price","features")
                .withColumnRenamed("price", "label");

//        modelInputData.show();

        Dataset<Row>[] trainingAndTestData =  modelInputData.randomSplit(new double[] {0.8,0.2});
        Dataset<Row> trainingData = trainingAndTestData[0];
        Dataset<Row> testData = trainingAndTestData[1];

        LinearRegressionModel model =  new LinearRegression().fit(trainingData);
        model.transform(testData).show();

    }
}
