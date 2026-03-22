using System.Diagnostics;

using RMV.ML.Network.Configuration;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Fully Connected Neural Network
/// </summary>
public class Net 
{

	#region Constructor --------------------------------------------------------	

	/// <summary>
	/// Builds a new Network with the specified activation function and settings.
	/// </summary>
	/// <param name="activation">The activation function for the hidden layers</param>
	/// <param name="settings">The application settings</param>
	public Net( AppSettings settings )
	{
		this.settings = settings;
		this.InputLayer = new InputLayer( settings.Input, settings );
		var lastLayer = InputLayer;

		foreach( int nodes in settings.Hidden )
		{
			var hiddenLayer = new HiddenLayer( nodes, settings, lastLayer );
			this.HiddenLayers.Add( hiddenLayer );
			lastLayer = hiddenLayer;
		}

		this.OutputLayer = new OutputLayer( settings.Output, settings, lastLayer );
	}
	//public Net( AppSettings settings ) 
	//{	
	//	this.settings = settings;
	//	this.InputLayer = new Layer( settings.Input, LayerType.Input, settings );      
	//     var lastLayer = InputLayer;

	//	foreach( int nodes in settings.Hidden )
	//	{
	//		var hiddenLayer = new Layer( nodes, LayerType.Hidden, settings, lastLayer );
	//		this.HiddenLayers.Add( hiddenLayer );
	//		lastLayer = hiddenLayer;
	//	}		

	//	this.OutputLayer = new Layer( settings.Output, LayerType.Output, settings, lastLayer );		
	//}


	readonly AppSettings settings;
	readonly BaseLayer InputLayer, OutputLayer; 
   readonly List<BaseLayer> HiddenLayers = [];	
	//double Error => this.OutputLayer.Error; // Current Error in the network

	#endregion


	#region Properties ---------------------------------------------------------

	/// <summary>
	/// Learning type
	/// </summary>
	public LearningType Learning { get; set; }   

	#endregion


	#region Connect/Initialise -------------------------------------------------	

	/// <summary>
	/// Connects neurons of each layer to the next one 
	/// </summary>
	public void Connect()
	{		
		this.InputLayer.Connect();
				
		foreach( var hidden in this.HiddenLayers ) hidden.Connect();	
	}


	/// <summary>
	/// Nodes weight initialization
	/// </summary>
	public void Initialize()
	{					
		foreach( var hidden in this.HiddenLayers ) hidden.Initialize();

		OutputLayer.Initialize();
	}
	#endregion


	#region Training -----------------------------------------------------------

	/// <summary>
	/// Trains the model using the specified training and test data sets over a series of epochs.
	/// </summary>	
	/// <param name="trainSet">Training data set</param>
	/// <param name="testSet">Test data set</param>
	public async Task TrainSgd( DataSet trainSet, DataSet testSet, Stopwatch timer )
	{
		for( int epoch = 0; epoch < settings.Iterations; epoch++ )
		{
			double batchError = 0;

			for( int i = 0; i < settings.Batch; i++ )
			{
				TrainingBatch( trainSet );

				if( Learning == LearningType.Online ) Update( 1 );

				batchError += ((OutputLayer)OutputLayer).Error;
			}

			if( Learning == LearningType.Batch ) Update( settings.Batch );

			double averageError = batchError / settings.Batch;

			if( epoch % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}" );
				continue;
			}

			double accuracy = Testing( testSet );			

			await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Accuracy={accuracy:f4}" );			
		}
	}

	/// <summary>
	/// Trains the Net with a random pattern from the specified dataset.
	/// </summary>
	/// <param name="data">The dataset to train on</param>
	void TrainingBatch( DataSet data )
	{
		int index = Random.Shared.Next( data.Source.Count ); // random pattern index

		Forward( data.Source[ index ] );

		Backward( data.Target[ index ] );
	}

	/// <summary>
	/// Trains the model using mini-batch gradient descent for a specified number of iterations	
	/// </summary>	
	/// <param name="trainSet">The training data set used to generate random mini-batches for each epoch</param>
	/// <param name="testSet">The data set used for validation testing at configured intervals</param>
	/// <param name="timer">A stopwatch instance used to track and report elapsed training time</param>
	/// <returns>A task that represents the asynchronous training operation</returns>
	public async Task TrainMiniBatch( DataSet trainSet, DataSet testSet, Stopwatch timer )
	{
		for( int iteration = 0; iteration < settings.Iterations; iteration++ )
		{
			#region reserved
			//var trainTask = Task.Run( () => trainSet.GetRandomBatch( settings.Batch ) );
			//var testTask = Task.Run( () => trainSet.GetRandomBatch( settings.Batch ) );			
			//Task.WaitAll( trainTask, testTask ); // Wait for both to finish
			//double averageError = TrainingMiniBatch( trainTask.Result ); // mini-batch train and get the average error
			#endregion

			var trainBatch = trainSet.GetRandomBatch( settings.Batch );
			var testBatch = testSet.GetRandomBatch( settings.Batch );		

			double averageError = TrainingMiniBatch( trainBatch );

			Update( settings.Batch ); // Update weights after processing the mini-batch

			if( iteration % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				await ReportAsync( $"Iteration={iteration}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}" );				
			}
			else  // Test every configured number of epochs and report accuracy			
			{
				double accuracy = Testing( testBatch );  // Test the mini-batch and get accuracy
				await ReportAsync( $"Iteration={iteration}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Accuracy={accuracy:f4}" );				
			}

			if( iteration % settings.Epoch == 0 ) // Perform validation testing at configured intervals
			{
				double accuracy = Testing( testSet );
				await ReportAsync( $"Iteration={iteration}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Validation={accuracy:f4}" );
			}
		}
		
		await ReportAsync( $"Time={timer.Elapsed:hh\\:mm\\:ss}  Validation={Testing( testSet ):f4}" );
	}

	/// <summary>
	/// Trains the Net with a random pattern from the specified dataset.
	/// </summary>
	/// <param name="data">The dataset to train on</param>
	/// <returns>The average error for the mini-batch</returns>
	double TrainingMiniBatch( DataSet data )
	{
		double error = 0;

		for( int i = 0; i < data.Source.Count; i++ )
		{
			Forward( data.Source[ i ] );
			Backward( data.Target[ i ] );
			error += ((OutputLayer)OutputLayer).Error;
		}

		return error / data.Source.Count;
	}

	#region reserved
	/// <summary>
	/// Trains the model using the dataflow pipelinefor a specified number of iterations	
	/// </summary>	
	/// <param name="trainSet">The dataset used for training the model. Must contain sufficient data for batch processing.</param>
	/// <param name="testSet">The dataset used for validation testing. Used to evaluate model accuracy at specified intervals.</param>
	/// <param name="timer">A stopwatch instance used to track and report elapsed training time.</param>
	/// <returns>A task that represents the asynchronous training operation.</returns>
	//public async Task TrainDataflow( DataSet trainSet, DataSet testSet, Stopwatch timer )
	//{
	//	var buffer = new BufferBlock<(double[], double[])>();
	//	var consumerTask = ConsumeAsync( buffer );

	//	for( int iteration = 0; iteration < settings.Iterations; iteration++ )
	//	{
	//		trainSet.Produce( settings.Batch, buffer ); // Asynchronously produce a batch of training data into the buffer			

	//		double averageError = TrainingMiniBatch( trainBatch );

	//		Update( settings.Batch );

	//		if( iteration % settings.Print != 0 ) // Don't test every epoch, just display error
	//		{
	//			await ReportAsync( $"Iteration={iteration}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}" );
	//		}
	//		else  // Test every configured number of epochs and report accuracy			
	//		{
	//			var testBatch = testSet.GetRandomBatch( settings.Batch );
	//			double accuracy = Testing( testBatch );
	//			await ReportAsync( $"Iteration={iteration}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Accuracy={accuracy:f4}" );
	//		}

	//		if( iteration % settings.Epoch == 0 ) // Perform validation testing at configured intervals
	//		{
	//			double accuracy = Testing( testSet );
	//			await ReportAsync( $"Iteration={iteration}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Validation={accuracy:f4}" );
	//		}
	//	}
	//}

	//async Task<double> ConsumeAsync( ISourceBlock<(double[], double[])> source )
	//{
	//	double error = 0;

	//	while( await source.OutputAvailableAsync() )
	//	{
	//		var data = await source.ReceiveAsync();
	//		Forward( data.Item1 );
	//		Back( data.Item2 );
	//		error += this.Error;
	//	}

	//	return error;
	//}	

	//double ParallelTrainingMiniBatch( DataSet data )
	//{
	//	object lockObject = new();
	//	double totalError = 0;
	//	int count = data.Source.Count;

	//	Parallel.For<double>(
	//	 0, count,
	//	 // 1. Thread Local Storage: Initialize local error for each thread
	//	 () => 0.0,
	//	 // 2. Loop Body: Accumulate error for each thread
	//	 ( i, state, localError ) => 
	//	 {
	//		 var localNet = this.Clone(); // Create a clone of 'this' for each thread
	//		 localNet!.Forward( data.Source[ i ] );
	//		 localNet.Back( data.Target[ i ] );
	//		 return localError + localNet.Error; // Accumulate local error
	//	 },

	//	 // 3. Local Finally: Safely merge the local error into the total error
	//	 localError => {
	//		 lock( lockObject )
	//		 {
	//			 totalError += localError;	 // You would also merge gradients/weights here if needed
	//		 }
	//	 }
	//	);

	//	return totalError / count;
	//}
	#endregion

	/// <summary>
	/// Feed forward through the network with the specified input pattern.
	/// </summary>
	/// <param name="input"></param>
	void Forward( double[] input )
	{
		this.InputLayer.Forward( input );

		foreach( var hidden in this.HiddenLayers ) hidden.Forward();

		this.OutputLayer.Forward();
	}

	/// <summary>
	/// Backpropagation of the error through the network with the specified target pattern.
	/// </summary>
	/// <param name="target"></param>
	void Backward( double[] target )
	{
		this.OutputLayer.Backward( target );

		HiddenLayers.Reverse();
		HiddenLayers.ForEach( h => h.Backward() );
		HiddenLayers.Reverse();		
	}	

	/// <summary>
	/// Update weights 
	/// </summary>
	public void Update( int batchSize )
	{	
		foreach( var hidden in this.HiddenLayers ) hidden.Update( batchSize );

		this.OutputLayer.Update( batchSize );
	}

	#endregion


	#region Testing ------------------------------------------------------------	

	/// <summary>
	/// Test network against dataset
	/// </summary>
	/// <param name="testData">The dataset to test against</param>
	/// <returns>The accuracy of the network on the test dataset</returns>
	public float Testing( DataSet testData )
	{
		var count = 0;

		for( int index = 0; index < testData.Source.Count; index++ )
		{
			Forward( testData.Source[ index ] );

			if( testData.Match( index, ((OutputLayer)OutputLayer).GetMaxItemIndex() ) ) count++;
		}

		return (float)count / testData.Source.Count;
	}

	#endregion	


	#region Report ---------------------------------------------------------------

	public EventHandler<ReportEventArgs> OnReport;	

	protected async Task ReportAsync( string message ) => await TriggerReportAsync( new ReportEventArgs { Message = message } );

	async Task TriggerReportAsync( ReportEventArgs ea ) => await Task.Run( () => OnReport.Invoke( null, ea ) );

	public class  ReportEventArgs : EventArgs
	{
		public required string Message 
		{
			get { return this.message.TrimEnd('\r', '\n'); }
			set { this.message = value; } 
		}
		string message = string.Empty;
	}

	#endregion
}

