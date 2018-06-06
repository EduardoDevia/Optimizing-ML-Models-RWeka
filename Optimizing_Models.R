Tunning_parameters<-function(DataFrame,Split_Value,Size,SMOTE,Model,Model_to_tune){
  
  
  
  #==================Resampling data ===========================================
  #Load weka filter
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  #Resample the data with the Size given
  DataFrame<-resample(target~ .,data=DataFrame,control=Weka_control(Z=Size))
  
  #==================Normalization ============================================
  #Select the numberic columns to be normalized
  columns_to_change<-colnames(select_if(DataFrame, is.numeric))
  #Normalize Dataframe
  for(i in 1:length(columns_to_change)){column_number<-which(colnames(DataFrame)==columns_to_change[i])  
  DataFrame[,column_number]<-data.Normalization (DataFrame[,column_number] ,type="n1",normalization="column")}
  
  #==================Split data ==============================================
  #Create a new dataframe
  data=DataFrame
  #Create a split data with a split feature
  split = sample.split(data$target, SplitRatio = Split_Value)
  #Create training set
  training_set = subset(data, split == TRUE)
  #Create testing set
  test_set = subset(data, split == FALSE)
  
  #==================Split data ==============================================
  
  if (SMOTE=='Y') {
    #str(training_set$target)
    prop.table(table(training_set$target))
    training_set<-SMOTE(target ~ ., training_set, perc.over = 100, perc.under=200)
    #prop.table(table(training_set$target))
    
  }
  
  
  
  #Create a dataframe to record the optimal size of the dataframe
  Optimal_Tunning<-data.frame(Seed=double(),Depth=double(),Neighbour=double(),confidence=double(),
                              minObjleaf=double(),MinInstances=double(),
                              Accuracy=double(),
                              CV_Accuracy=double(),Test_Accuracy=double(),
                              "Acc-Test"=double(),"Acc-Acc_CV"=double())
  
  
  
  #===============================================================================
  #==================================RandomForest=================================
  #===============================================================================
  
  
  if (Model="RandomForest"){
    
    
    
    #Find the most optimal size of the file that reduces the overfitting.
    #"p" represent the maximum depth of the tree and "i" the number of random seed number for selecting
    #attributes
    
    for (p in 1:10) {
      
      for(i in 1:4) {
        
        #Run the RandomForest
        RandomForest_Classifier<-Model_to_tune
        
        Train<-summary(RandomForest_Classifier)
        Accuracy<-Train$details[1]
        #Cross Validation
        CV <- evaluate_Weka_classifier(RandomForest_Classifier, numFolds = 10,
                                                    complexity = FALSE, seed = 1, class = TRUE)
        #Accuracy CV
        CV_Accuracy<-CV$details[1]
        Test<-table( predict(RandomForest_Classifier,newdata=test_set),
                                  test_set$target )
        
        #Record Depth and Seed
        Depth<-p
        Seed<-i
        
        #Accuracy Test
        Test_Accuracy<-(Test[1,1]+Test[2,2])/(Test[1,1]+Test[2,2]+Test[2,1]+Test[2,1])*10
        
        #Record the results in the Dataframe
        Optimal_Tunning[nrow(Optimal_Tunning) + 1,] <-list(
          Seed=Seed,Depth=Depth,Neighbour=Neighbour,Accuracy=Accuracy,
          CV_Accuracy=CV_Accuracy,Test_Accuracy=Test_Accuracy,
          "Acc-Test"=Accuracy-Test_Accuracy,
          "Acc-Acc_CV"=Accuracy-CV_Accuracy)
      }
    }
    
  
  }
  
  #===============================================================================
  #==========================================IBk==================================
  #===============================================================================
  
  
  if (Model="IBk"){
    
  
    
    for (i in 1:20) {
      Neighbour=i
      IBk_Classifier<-IBk(train_file_clean$target~ ., data = train_file_clean,control=Weka_control(K=Neighbour))
      Train<-summary(IBk_Classifier)
      Accuracy<-Train$details[[1]]
      
      #Cross Validation
      CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
      #Cross Validation
      CV_Accuracy<-CV$details[[1]]
      
      Test<-table( predict(IBk_Classifier,newdata=test_set),test_set$target )
      
     #Accuracy Test
      Test_Accuracy<-(Test[1,1]+Test[2,2])/(Test[1,1]+Test[2,2]+Test[2,1]+Test[2,1])*10
     
      
      Optimal_Tunning[nrow(Optimal_Tunning) + 1,] <-list(
        Seed=Seed,Depth=Depth,Neighbour=Neighbour,Accuracy=Accuracy,
        CV_Accuracy=CV_Accuracy,Test_Accuracy=Test_Accuracy,
        "Acc-Test"=Accuracy-Test_Accuracy,
        "Acc-Acc_CV"=Accuracy-CV_Accuracy)
    }
    
  }
  
  #===============================================================================
  #==========================================J48==================================
  #===============================================================================
  
  if (Model="J48"){
    
    #===============J48 C & M Optimisation===================
    for (i in 1:100) { confidence<-runif(1, 0, 0.25) minObjleaf<-floor(runif(1, 1,10)) 
      
      J48_Classifier<-J48(train_file_clean$target~ ., data = train_file_clean,
                          control=Weka_control(C=confidence,M=minObjleaf))
      
      Train<-summary(J48_Classifier)
      Accuracy<-Train$details[[1]]
      
      #Cross Validation
      CV <- evaluate_Weka_classifier(J48_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
      CV_Accuracy<-CV$details[[1]]
      
      
      Test<-table( predict(J48_Classifier,newdata=test_set),test_set$target )
      
      #Accuracy Test
      Test_Accuracy<-(Test[1,1]+Test[2,2])/(Test[1,1]+Test[2,2]+Test[2,1]+Test[2,1])*10
      
      
    }
    
  }
  
  #===============================================================================
  #==========================================M5P==================================
  #===============================================================================
  
  
  if (Model="M5P"){
    for (i in 1:50) {  MinInstances<-i+1
    
      M5P_Classifier<-M5P(train_file_clean$age~ ., data = train_file_clean, control=Weka_control(M=MinInstances,U=FALSE))
      Train<-summary(M5P_Classifier)
      Accuracy<-Train$details[[1]]
      
      #Cross Validation
      CV <- evaluate_Weka_classifier(M5P_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
      #Cross Validation
      CV_Accuracy<-CV$details[[1]]
      Test<-table( predict(J48_Classifier,newdata=test_set),test_set$target )
      
      #Accuracy Test
      Test_Accuracy<-(Test[1,1]+Test[2,2])/(Test[1,1]+Test[2,2]+Test[2,1]+Test[2,1])*10
    }  
      
  }
  
  
  return(Optimal_Tunning)  
  
  
  
}