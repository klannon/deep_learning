----------------------------------------------------------------------
-- This script demonstrates how to load the Face Detector 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet, Eugenio Culurciello
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'torch'   -- torch
csvData = {}


-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

function readFile(filePath)
        print("in readfile ")
-- Read data from CSV to tensor
	local csvFile = io.open(filePath, 'r')  
	local header = csvFile:read()

-- Count number of rows and columns in file
	local i = 0
	for line in io.lines(filePath) do
	  if i == 0 then
	    COLS = #line:split(',')
	  end
	  i = i + 1
	end
	
	local ROWS = i - 1  -- Minus 1 because of header
	
	local data = torch.Tensor(ROWS, COLS)

	local i = 0  
	for line in csvFile:lines('*l') do  
	  i = i + 1
	  local l = line:split(',')
          count = 1
	  for key, val in ipairs(l) do
	    data[i][count] = val
            if (i<10) then
	       --if (count <20) then
                  print("i: "..count.."; val: "..val)
               --end
               if (count == 758) then
                  print("i: "..i.."; label: "..val)
	       end
            end
            count = count + 1	

	  end
	end
	csvFile:close()  
	return data,ROWS,COLS
end

function csvData.getData(pathTrain,pathTest)
	testD,testROWS,testCOLS = readFile(pathTest)
	trainD,trainROWS,trainCOLS = readFile(pathTrain)
        print("test rows "..testROWS.."; cols "..testCOLS)
        print("train rows "..trainROWS.."; cols "..trainCOLS)
        
   labelsAll =  torch.Tensor(trainROWS)
	labelsAll = trainD[{ {},1}]:clone()
   trainAll =  torch.Tensor(trainROWS, trainCOLS-1)
	trainAll = trainD[{ {},{2,trainCOLS}}]:clone()
	for i=1,10 do	
            print("label "..labelsAll[i])
 	    for k=1,5 do
	           print("train "..trainAll[i][k])
	    end
	end

   testAll =  torch.Tensor(testROWS, testCOLS-1)
	testAll = testD[{ {},{2,testCOLS}}]:clone()

-- shuffle dataset: get shuffled indices in this variable:
	local labelsShuffle = torch.randperm((#labelsAll)[1])
	print(labelsShuffle:size())
	local portionTrain = 0.8 -- 80% is train data, rest is test data
	local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
	local tesize = labelsShuffle:size(1) - trsize

	print("trsize  "..trsize.."; tesize "..tesize)
	print("trwidth "..trainAll:size(2).."; tewdith "..testAll:size(2))
        print("test rows "..testROWS.."; cols "..testCOLS)
        print("train rows "..trainROWS.."; cols "..trainCOLS)

-- create train set:
	local trainData = {
	   data = torch.Tensor(trsize, trainCOLS-1),
	   labels = torch.Tensor(trsize),
	   size = function() return trsize end
	}
--create val set:
	local valData = {
	    data = torch.Tensor(tesize, trainCOLS-1),
	    labels = torch.Tensor(tesize),
 	    size = function() return tesize end
   	}
--create test set:
	local testData = {
	    data = torch.Tensor(testROWS, testCOLS-1),
 	    size = function() return testROWS end
   	}

	for i=1,trsize do
	   trainData.data[i] = trainAll[{ labelsShuffle[i],{} }]:clone()
	   trainData.labels[i] = labelsAll[labelsShuffle[i]]
	   if (i<10) then
	      print("row: "..i.."; label "..trainData.labels[i])
	      for k=1,5 do
		print(trainData.data[{i,k}])
	      end
	   end
	end
	for i=trsize+1,tesize+trsize do
	   valData.data[i-trsize] = trainAll[{ labelsShuffle[i],{} }]:clone()
	   valData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
	end

	--print(trainData.data[1])

-- Exports
	return trainData,valData,testD

end

