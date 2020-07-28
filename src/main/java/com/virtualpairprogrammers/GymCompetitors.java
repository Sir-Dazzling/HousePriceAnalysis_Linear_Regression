package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GymCompetitors
{
    public static void main(String[] args)
    {
        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark =  SparkSession.builder()
                .appName("GymCompetitors").master("local[*]")
                .config("spark.sql.warehouse.dir","file:///c:/tmp/")
                .getOrCreate();


        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/GymCompetition.csv");

        //csvData.printSchema();
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[] {"Age","Height","Weight"});
        vectorAssembler.setOutputCol("features");
        Dataset<Row> csvDataWithFeatures = vectorAssembler.transform(csvData);

        Dataset<Row> modelInputData = csvDataWithFeatures.select("NoOfReps","features").withColumnRenamed("NoOfReps", "label");
        modelInputData.show();

        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel model = linearRegression.fit(modelInputData);
        System.out.println("The model has intercept " + model.intercept() + " and coefficients " + model.coefficients());

        model.transform(modelInputData).show();

    }
}