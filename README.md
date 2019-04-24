# dmc2019
Data Mining Cup 2019

### Описание исходных фичей:

- trustLevel (уровень доверия) - {1,2,3,4,5,6}. 6 - наибольший уровень доверия

- totalScanTimeInSeconds (общее время сканирования в секундах) - время от первого до последнего отсканированного продукта

- grandTotal (итого) - итоговая сумма по покупке

- lineItemVoids - количество отмененных (непрошедших, несработавших, ошибочных) сканирований

- scansWithoutRegistration - количество попыток активировать сканер (видимо нажать на кнопку) впустую, не сканируя ничего

- quantityModification - количество модификаций числа одного отсканированного продукта

- scannedLineItemsPerSecond - среднее число сканированных продуктов в секунду (частота)

- valuePerSecond - среднее число в деньгах, проходящее через кассу в секунду (grandTotal/totalScanTimeInSeconds)

- lineItemVoidsPerPosition - среднее число ошибочных сканирований, поделенное на общее число безошибочных сканирований (lineItemVoids/scannedLineItemsPerSecond*totalScanTimeInSeconds)

### Описание искусственных фичей (из репозиториев):

- totalScanned (общее число сканов за транзакцию) = scannedLineItemsPerSecond*totalScanTimeInSeconds

- avgTimePerScan (среднее время сканирования, период) = 1/scannedLineItemsPerSecond

- avgValuePerScan (среднее значение в деньгах на скан) = avgTimePerScan*valuePerSecond

- withoutRegisPerPosition (??) = scansWithoutRegistration*totalScanned

- quantityModsPerPosition (количество модификаций кол-ва к общему числу сканов) = quantityModifications/totalScanned

- lineItemVoidsPerPosition (количество отмен к общему числу сканов) = lineItemVoids/totalScanned

- !! lineItemVoidsPerTotal (количество отмен к итоговой сумме) = lineItemVoids/grandTotal

- !! withoutRegistrationPerTotal (количество пустых сканов к итоговой сумме) = scansWithoutRegistration/grandTotal

- !! quantiModsPerTotal (число модификаций кол-ва к итоговой сумме) = quantityModifications/grandTotal

- !! lineItemVoidsPerTime (число отмен к общему времени сканирования) = lineItemVoids/totalScanTimeInSeconds

- !! withoutRegistrationPerTime (число пустых сканов к общему времени сканирования) = scansWithoutRegistration/totalScanTimeInSeconds 

- !! quantiModesPerTime (число модификаций кол-ва к общему времени сканирования) = quantityModifications/totalScanTimeInSeconds

### Описание искусственных фичей:


X['logTotalScanned'] = X['avgValuePerScan']/X['grandTotal']

#X['absAvgValuePerScanPerGrandTotal'] = np.abs(X['avgValuePerScan']-X['grandTotal'])