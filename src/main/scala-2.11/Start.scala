/**
  * Created by samay on 8/16/16.
  */
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, SparkSession}


object Start extends App {
  // setup spark
  private val talkingDataBasePath = "/home/samay/Projects/TalkingDataFiles/"
  val spark = SparkSession.builder().master("local").appName("TalkingData").config("spark.sql.pivotMaxValues",10L).getOrCreate()
  val sc = spark.sparkContext
  import spark.implicits._

  // Hajime!
  val in = GetRawData
  val labeledData = spark.read.option("header", true).csv(talkingDataBasePath + "gender_age_train.csv").as[Target]


  val distinctLabels = in.labelCategories.select("label_id").map(_(0).toString.toLong).collect()

  val eventCountForDeviceLabel =
    in.events
      .join(in.appEvents, "event_id")
      .join(in.appLabels, "app_id")
      .groupBy("device_id")
      .pivot("label_id", distinctLabels)
      .count()

  var allFeatures = eventCountForDeviceLabel.join(in.deviceInfo, "device_id").join(labeledData, "device_id").na.fill(0)

  val columns = (eventCountForDeviceLabel.columns.toList.drop(1)).toArray
  val putFeaturesInAVector =
    new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")
      .transform(allFeatures)
      .select("group", "features")

  val groupIndexed = new StringIndexer().setInputCol("group").setOutputCol("group_id").fit(labeledData).transform(labeledData).join(putFeaturesInAVector, "group").select("group_id","features")
  val mllibFormatted = MLUtils.convertVectorColumnsFromML(groupIndexed, "features").map(row => new LabeledPoint(row.getAs[Double]("group_id"), row.getAs[mllib.linalg.Vector]("features"))).rdd
  val Array(trainingData, testData) = mllibFormatted.randomSplit(Array(0.7, 0.3))

  val model = new LogisticRegressionWithLBFGS()
    .setNumClasses(12)
    .run(trainingData)

  val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }

  // Get evaluation metrics.
  val metrics = new MulticlassMetrics(predictionAndLabels)
  val accuracy = metrics.accuracy
  println(s"Accuracy = $accuracy")

//
//  val rf = new RandomForestClassifier()
//    .setLabelCol("group_id")
//    .setFeaturesCol("features")
//    .setNumTrees(10)
//
//  val model = rf.fit(trainingData)
//
//  val predictions = model.transform(testData)
//
//  val evaluator = new MulticlassClassificationEvaluator()
//    .setLabelCol("group_id")
//    .setPredictionCol("prediction")
//    .setMetricName("accuracy")
//
//  val accuracy = evaluator.evaluate(predictions)
//  println("Test Error = " + (1.0 - accuracy))
//
//  model.save("/home/samay/Projects/TalkingDataLogisticRegression")

  def GetRawData: RawData = {
    val appEvents = spark.read.option("header", "true").csv(talkingDataBasePath + "app_events.csv")
      .as[AppEvent]
    val appLabels = spark.read.option("header", "true").csv(talkingDataBasePath + "app_labels.csv")
      .as[AppLabel]
    val events = spark.read.option("header", "true").csv(talkingDataBasePath + "events.csv")
      .map(r => Event(r(0).toString.toLong, r(1).toString.toLong))
    val labelCategories = spark.read.option("header", "true").csv(talkingDataBasePath + "label_categories.csv")
      .as[LabelCategory]
    val deviceInfo = spark.read.option("header", "true").csv(talkingDataBasePath + "phone_brand_device_model.csv")
      .as[Device].dropDuplicates("device_id")

    return RawData(events, appEvents, appLabels, labelCategories, deviceInfo)
  }
}
