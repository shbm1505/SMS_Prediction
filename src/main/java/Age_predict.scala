import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{StructType, StructField, StringType,IntegerType};
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import  org.apache.spark.ml.util.MLWritable
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel

object Age_predict {
   def main(args:Array[String])
  {
  
  val conf=new SparkConf()
    .setAppName("SmsPrediction")
    
    val sc = new SparkContext(conf)
  
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  
 val customSchema = StructType(Array(StructField("phoneno", StringType, true),
StructField("operator", StringType, true),
StructField("circle", StringType, true),
StructField("gender", StringType, true),StructField("gender_p", StringType, true),StructField("gender_id", StringType, true),StructField("gender_date", StringType, true),StructField("gender_fromnumber", StringType, true),
StructField("location", StringType, true),StructField("location_p", StringType, true),StructField("location_id", StringType, true),StructField("location_date", StringType, true),StructField("location_fromnumber", StringType, true),
StructField("health_insurance ", StringType, true),StructField("health_insurance_p ", StringType, true),StructField("health_insurance_id ", StringType, true),StructField("health_insurance_date ", StringType, true),StructField("health_insurance_fromnumber ", StringType, true),
StructField("mutual_fund", StringType, true),StructField("mutual_fund_p", StringType, true),StructField("mutual_fund_id", StringType, true),StructField("mutual_fund_date", StringType, true),StructField("mutual_fund_fromnumber", StringType, true),
StructField("saving_account", StringType, true),StructField("saving_account_p", StringType, true),StructField("saving_account_id", StringType, true),StructField("saving_account_date", StringType, true),StructField("saving_account_fromnumber", StringType, true),
StructField("kids", StringType, true),StructField("kids_p", StringType, true),StructField("kids_id", StringType, true),StructField("kids_date", StringType, true),StructField("kids_fromnumber", StringType, true),
StructField("credit_card", StringType, true),StructField("credit_card_p", StringType, true),StructField("credit_card_id", StringType, true),StructField("credit_card_date", StringType, true),StructField("credit_card_fromnumber", StringType, true),
StructField("home_loan", StringType, true),StructField("home_loan_p", StringType, true),StructField("home_loan_id", StringType, true),StructField("home_loan_date", StringType, true),StructField("home_loan_fromnumber", StringType, true),
StructField("car_insurance", StringType, true),StructField("car_insurance_p", StringType, true),StructField("car_insurance_id", StringType, true),StructField("car_insurance_date", StringType, true),StructField("car_insurance_fromnumber", StringType, true),
StructField("car_loan", StringType, true),StructField("car_loan_p", StringType, true),StructField("car_loan_id", StringType, true),StructField("car_loan_date", StringType, true),StructField("car_loan_fromnumber", StringType, true),
StructField("life_insurance", StringType, true),StructField("life_insurance_p", StringType, true),StructField("life_insurance_id", StringType, true),StructField("life_insurance_date", StringType, true),StructField("life_insurance_fromnumber", StringType, true),
StructField("age", StringType, true),StructField("age_p", StringType, true),StructField("age_id", StringType, true),StructField("age_date", StringType, true),StructField("age_fromnumber", StringType, true),
StructField("income", StringType, true),StructField("income_p", StringType, true),StructField("income_id", StringType, true),StructField("income_date", StringType, true),StructField("income_fromnumber", StringType, true),
StructField("net_banking", StringType, true),StructField("net_banking_p", StringType, true),StructField("net_banking_id", StringType, true),StructField("net_banking_date", StringType, true),StructField("net_banking_fromnumber", StringType, true),
StructField("travelcount", StringType, true),StructField("travelcount_p", StringType, true),StructField("travelcount_id", StringType, true),StructField("travelcount_date", StringType, true),StructField("travelcount_fromnumber", StringType, true),
StructField("shoppercount", StringType, true),StructField("shoppercount_p", StringType, true),StructField("shoppercount_id", StringType, true),StructField("shoppercount_date", StringType, true),StructField("shoppercount_fromnumber", StringType, true),
StructField("creditcount", StringType, true),StructField("creditcount_p", StringType, true),StructField("creditcount_id", StringType, true),StructField("creditcount_date", StringType, true),StructField("creditcount_fromnumber", StringType, true),
StructField("withdrawlcount", StringType, true),StructField("withdrawlcount_p", StringType, true),StructField("withdrawlcount_id", StringType, true),StructField("withdrawlcount_date", StringType, true),StructField("withdrawlcount_fromnumber", StringType, true),
StructField("withdrawlsum", StringType, true),StructField("withdrawlsum_p", StringType, true),StructField("withdrawlsum_id", StringType, true),StructField("withdrawlsum_date", StringType, true),StructField("withdrawlsum_fromnumber", StringType, true),
StructField("creditsum", StringType, true),StructField("creditsum_p", StringType, true),StructField("creditsum_id", StringType, true),StructField("creditsum_date", StringType, true),StructField("creditsum_fromnumber", StringType, true),
StructField("e_expense", StringType, true),StructField("e_expense_p", StringType, true),StructField("e_expense_id", StringType, true),StructField("e_expense_date", StringType, true),StructField("e_expense_fromnumber", StringType, true),
StructField("e_count", StringType, true),StructField("e_count_p", StringType, true),StructField("e_count_id", StringType, true),StructField("e_count_date", StringType, true),StructField("e_count_fromnumber", StringType, true),
StructField("avg_brand_count", StringType, true),StructField("avg_brand_count_p", StringType, true),StructField("avg_brand_count_id", StringType, true),StructField("avg_brand_count_date", StringType, true),StructField("avg_brand_count_fromnumber", StringType, true),
StructField("premium_brand_count", StringType, true),StructField("premium_brand_count_p", StringType, true),StructField("premium_brand_count_id", StringType, true),StructField("premium_brand_count_date", StringType, true),StructField("premium_brand_count_fromnumber", StringType, true),
StructField("apparels", StringType, true),StructField("apparels_p", StringType, true),StructField("apparels_id", StringType, true),StructField("apparels_date", StringType, true),StructField("apparels_fromnumber", StringType, true),
StructField("gadgets", StringType, true),StructField("gadgets_p", StringType, true),StructField("gadgets_id", StringType, true),StructField("gadgets_date", StringType, true),StructField("gadgets_fromnumber", StringType, true),
StructField("sports", StringType, true),StructField("sports_p", StringType, true),StructField("sports_id", StringType, true),StructField("sports_date", StringType, true),StructField("sports_fromnumber", StringType, true),
StructField("movies", StringType, true),StructField("movies_p", StringType, true),StructField("movies_id", StringType, true),StructField("movies_date", StringType, true),StructField("movies_fromnumber", StringType, true),
StructField("dob", StringType, true),StructField("dob_p", StringType, true),StructField("dob_id", StringType, true),StructField("dob_date", StringType, true),StructField("dob_fromnumber", StringType, true),
StructField("homeloanemi", StringType, true),StructField("homeloanemi_p", StringType, true),StructField("homeloanemi_id", StringType, true),StructField("homeloanemi_date", StringType, true),StructField("homeloanemi_fromnumber", StringType, true),
StructField("homeloanemiduedate", StringType, true),StructField("homeloanemiduedate_p", StringType, true),StructField("homeloanemiduedate_id", StringType, true),StructField("homeloanemiduedate_date", StringType, true),StructField("homeloanemiduedate_fromnumber", StringType, true),
StructField("carloan", StringType, true),StructField("carloan_p", StringType, true),StructField("carloan_id", StringType, true),StructField("carloan_date", StringType, true),StructField("carloan_fromnumber", StringType, true),StructField("carloanduedate", StringType, true),
StructField("healthinsuranceemi", StringType, true),StructField("healthinsuranceemi_p", StringType, true),StructField("healthinsuranceemi_id", StringType, true),StructField("healthinsuranceemi_date", StringType, true),StructField("healthinsuranceemi_fromnumber", StringType, true),
StructField("healthinsuranceemidue", StringType, true),StructField("healthinsuranceemidue_p", StringType, true),StructField("healthinsuranceemidue_id", StringType, true),StructField("healthinsuranceemidue_date", StringType, true),StructField("healthinsuranceemidue_fromnumber", StringType, true),
StructField("lifeinsuranceemi", StringType, true),StructField("lifeinsuranceemi_p", StringType, true),StructField("lifeinsuranceemi_id", StringType, true),StructField("lifeinsuranceemi_date", StringType, true),StructField("lifeinsuranceemi_fromnumber", StringType, true),
StructField("lifeinsuranceemidue", StringType, true),StructField("lifeinsuranceemidue_p", StringType, true),StructField("lifeinsuranceemidue_id", StringType, true),StructField("lifeinsuranceemidue_date", StringType, true),StructField("lifeinsuranceemidue_fromnumber", StringType, true),
StructField("carinsuranceemi", StringType, true),StructField("carinsuranceemi_p", StringType, true),StructField("carinsuranceemi_id", StringType, true),StructField("carinsuranceemi_date", StringType, true),
StructField("carinsuranceemidue", StringType, true),StructField("carinsuranceemidue_p", StringType, true),StructField("carinsuranceemidue_id", StringType, true),StructField("carinsuranceemidue_date", StringType, true),StructField("carinsuranceemidue_fromnumber", StringType, true),
StructField("cclimit", StringType, true),StructField("cclimit_p", StringType, true),StructField("cclimit_id", StringType, true),StructField("cclimit_date", StringType, true),StructField("cclimit_fromnumber", StringType, true),
StructField("name", StringType, true),StructField("name_p", StringType, true),StructField("name_id", StringType, true),StructField("name_date", StringType, true),StructField("name_fromnumber", StringType, true)))
 
  
  val input_data = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").option("delimiter","|").schema(customSchema).load("hdfs://master:9000/user/hive/warehouse/userprofile/part-r-00000")
  
  val input_data2=input_data.select("phoneno","gender","creditcount","withdrawlsum","e_expense","avg_brand_count","premium_brand_count","income","location","age");
  val transforms = udf((gender: String) => {
  if (gender == "male") "0"
  else if(gender=="female") "1"
  else "999" 
})

 val transforms5 = udf((gender: String) => {
  if (gender == "NA") "0"
  
  else "0" 
})
 val transforms2 = udf((income: String) => {
  if (income == "0-3L") "1"
  else if(income=="3-6L") "2"
  else if(income=="6-10L") "3"
  else if(income=="10-15L") "4"
  else if(income=="15-20L") "5"
  else if(income=="20-25L") "6"
  else if(income=="25-40L") "7"
  else if(income=="40-60L") "8"
    else if(income=="60L-1CR") "9"
      else if(income=="1CR+") "10"
  else "999" 
})

val transforms3 = udf((age: String) => {
  if (age == "<18") "1"
  else if(age=="18-25") "2"
  else if(age=="25-30") "3"
  else if(age=="30-35") "4"
  else if(age=="35-40") "5"
  else if(age=="40-45") "6"
  else if(age=="45-50") "7"
  else if(age==">50") "8"
  else "NA" 
})
val transforms4 = udf((location: String) => {
  if (location == "hyderabad") "1"
  else if(location=="Delhi") "1"
  else if(location=="ahmedabad") "1"
  else if(location=="bangalore") "1"
  else if(location=="Mumbai") "1"
  else if(location=="mumbai") "1"
  else if(location=="delhi") "1"
  else if(location=="Kolkata") "1"
  else if (location == "kolkata") "1"
  else if(location=="pune") "1"
  else if(location=="chennai") "1"
  else if(location=="kozhikode") "2"
  else if(location=="kochi") "2"
  else if(location=="thiruvanathpuram") "2"
  else if(location=="gwalior") "2"
  else if(location=="indore") "2"
  else if(location=="bhopal") "2"
  else if(location=="jabalpur") "2"
  else if(location=="guwahati") "2"
  else if(location=="patna") "2"
  else if(location=="chandigarh") "2"
  else if(location=="durg") "2"
  else if(location=="bhilai") "2"
  else if (location == "raipur") "2"
  else if(location=="rajkot") "2"
  else if(location=="ramnagar") "2"
  else if(location=="vadodra") "2"
  else if(location=="surat") "2"
  else if(location=="faridabad") "2"
  else if(location=="srinagar") "2"
  else if(location=="jamshedpur") "2"
  else if(location=="ranchi") "2"
  else if(location=="dhanbad") "2"
  else if(location=="belgaum") "2"
  else if(location=="mangalore") "2"
  else if(location=="mysore") "2"
  else if(location=="amravati") "2"
  else if(location=="nagpur") "2"
  else if (location == "aurangabad") "2"
  else if(location=="nashik") "2"
  else if(location=="bhiwandi") "2"
  else if(location=="solapur") "2"
  else if(location=="kolhapur") "2"
  else if(location=="cuttak") "2"
  else if(location=="bhubaneshwar") "2"
  else if(location=="amritsar") "2"
  else if(location=="jalandhar") "2"
  else if(location=="ludhiana") "2"
  else if(location=="pondicherry") "2"
  else if(location=="bikaner") "2"
  else if(location=="jaipur") "2"
  else if(location=="jodhpur") "2"
  else if(location=="kota") "2"
  else if (location == "salem") "2"
  else if(location=="tiruppur") "2"
  else if(location=="coimbatore") "2"
  else if(location=="tiruchirappalli") "2"
  else if(location=="madurai") "2"
  else if(location=="moradabad") "2"
  else if(location=="meerut") "2"
  else if(location=="ghaziabad") "2"
  else if(location=="aligarh") "2"
  else if(location=="agra") "2"
  else if(location=="bareilly") "2"
  else if(location=="lucknow") "2"
  else if(location=="kanpur") "2"
  else if(location=="allahabad") "2"
  else if(location=="gorakhpur") "2"
  else if (location == "varanasi") "2"
  else if(location=="dehradun") "2"
  else if(location=="noida") "2"
  else if(location=="gurgaon") "2"
  else if(location=="asanol") "2"
  else "3" 
})

val new_df=input_data2.withColumn("gender2",transforms(input_data("gender"))).withColumn("income2",transforms2(input_data("income"))).withColumn("age2",transforms3(input_data("age"))).withColumn("location2",transforms4(input_data("location")))

val new_df5=new_df.select("age2","phoneno","creditcount","withdrawlsum","e_expense","avg_brand_count","premium_brand_count","income2","gender2","location2")
val toDouble = udf[Double, String]( _.toDouble)

new_df5.show()

val new_df6=new_df5.filter("age2='NA'")

new_df6.show()

val training = new_df6
.withColumn("gender2", toDouble(new_df5("gender2")))
.withColumn("creditcount", toDouble(new_df5("creditcount")))
.withColumn("withdrawlsum",  toDouble(new_df5("withdrawlsum")))
.withColumn("e_expense",      toDouble(new_df5("e_expense")))
.withColumn("avg_brand_count", toDouble(new_df5("avg_brand_count")))
.withColumn("premium_brand_count",  toDouble(new_df5("premium_brand_count")))
.withColumn("income2",  toDouble(new_df5("income2")))
.withColumn("age2",      toDouble(new_df5("age2"))) 
.withColumn("location2",      toDouble(new_df5("location2")))

val assembler = new VectorAssembler()
  .setInputCols(Array("creditcount","withdrawlsum","e_expense","avg_brand_count","premium_brand_count"))
  .setOutputCol("features")

  val data = assembler.transform(training)

val data2=data.withColumn("label",toDouble(data("age2")))


val data3=data2.select("features","phoneno")

val a = sc.objectFile[PipelineModel]("hdfs://master:9000/model/age_m6").first()
val predictions=a.transform(data3)

predictions.show()

val toIncome = udf((age: String) => {
  if (age == "1.0") "<18"
  else if(age=="2.0") "18-25"
  else if(age=="3.0") "25-30"
  else if(age=="4.0") "30-35"
  else if(age=="5.0") "35-40"
  else if(age=="6.0") "40-45"
  else if(age=="7.0") "45-50"
    else "50+"
  
   
})



val y=predictions.withColumn("predict",toIncome(predictions("predictedLabel")))
val x=y.select("phoneno","predict","probability")



x.show()

x.write
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .save("hdfs://master:9000/ml_output/agefinaldataa")

}

}



  
    
  



