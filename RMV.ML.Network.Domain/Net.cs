using System.Diagnostics;

using RMV.ML.Network.Common;

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
		this.InputLayer = new Layer( settings.Input, LayerType.Input, settings );      
      var lastLayer = InputLayer;

		foreach( int nodes in settings.Hidden )
		{
			var hiddenLayer = new Layer( nodes, LayerType.Hidden, settings, lastLayer );
			this.HiddenLayers.Add( hiddenLayer );
			lastLayer = hiddenLayer;
		}		

		this.OutputLayer = new Layer( settings.Output, LayerType.Output, settings, lastLayer );		
	}	   

	readonly AppSettings settings;
	readonly Layer InputLayer, OutputLayer; 
   readonly List<Layer> HiddenLayers = []; 

	#endregion


	#region Properties ---------------------------------------------------------

	/// <summary>
	/// Learning type
	/// </summary>
	public LearningType Learning { get; set; }

   /// <summary>
   /// Current Error in the network.
   /// </summary>
   public double Error => this.OutputLayer.Error;

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
	public async Task Train( DataSet trainSet, DataSet testSet, Stopwatch timer )
	{
		for( int epoch = 0; epoch < settings.Iterations; epoch++ )
		{
			double batchError = 0;

			for( int i = 0; i < settings.Batch; i++ )
			{
				TrainingBatch( trainSet );

				if( Learning == LearningType.Online ) Update( 1 );

				batchError += this.Error;
			}

			if( Learning == LearningType.Batch ) Update( settings.Batch );

			double averageError = batchError / settings.Batch;

			if( epoch % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}" );
				continue;
			}

			double accuracy = Testing( testSet );			

			await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Accuracy={accuracy:f2}%" );			
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

		Back( data.Target[ index ] );
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
		for( int epoch = 0; epoch < settings.Iterations; epoch++ )
		{
			var trainTask = Task.Run( () => trainSet.GetRandomBatch( settings.Batch ) );
			var testTask = Task.Run( () => trainSet.GetRandomBatch( settings.Batch ) );			
			Task.WaitAll( trainTask, testTask ); // Wait for both to finish

			double averageError = TrainingMiniBatch( trainTask.Result );

			Update( settings.Batch );

			if( epoch % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}" );				
			}
			else  // Test every configured number of epochs and report accuracy			
			{
				double accuracy = Testing( testTask.Result );
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Accuracy={accuracy:f2}%" );				
			}

			if( epoch % settings.Validate == 0 ) // Perform validation testing at configured intervals
			{
				double accuracy = Testing( testSet );
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Validation={accuracy:f2}%" );
			}
		}
	}

	/// <summary>
	/// Trains the model using the dataflow pipelinefor a specified number of iterations	
	/// </summary>	
	/// <param name="trainSet">The dataset used for training the model. Must contain sufficient data for batch processing.</param>
	/// <param name="testSet">The dataset used for validation testing. Used to evaluate model accuracy at specified intervals.</param>
	/// <param name="timer">A stopwatch instance used to track and report elapsed training time.</param>
	/// <returns>A task that represents the asynchronous training operation.</returns>
	public async Task TrainDataflow( DataSet trainSet, DataSet testSet, Stopwatch timer )
	{		
		for( int epoch = 0; epoch < settings.Iterations; epoch++ )
		{
			var trainTask = Task.Run( () => trainSet.GetRandomBatch( settings.Batch ) );
			var testTask = Task.Run( () => trainSet.GetRandomBatch( settings.Batch ) );
			Task.WaitAll( trainTask, testTask ); // Wait for both to finish			

			double averageError = TrainingMiniBatch( trainTask.Result );

			Update( settings.Batch );

			if( epoch % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}" );
			}
			else  // Test every configured number of epochs and report accuracy			
			{
				double accuracy = Testing( testTask.Result );
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Accuracy={accuracy:f2}%" );
			}

			if( epoch % settings.Validate == 0 ) // Perform validation testing at configured intervals
			{
				double accuracy = Testing( testSet );
				await ReportAsync( $"Epoch={epoch}, Avg Error={averageError:f3} Time={timer.Elapsed:hh\\:mm\\:ss}  Validation={accuracy:f2}%" );
			}
		}
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
			Back( data.Target[ i ] );
			error += this.Error;
		}

		return error / data.Source.Count;
	}

	#region reserved
	//double ParallelTrainingMiniBatch( DataSet data )
	//{
	//	object lockObject = new object();
	//	double totalError = 0;
	//	int count = data.Source.Count;

	//	Parallel.For(
	//		 0, count,
	//		 // 1. Thread Local Storage: Create a clone of 'this' for each thread
	//		 // This prevents threads from stomping on each other's internal state.
	//		 () => this.DeepCopy(),

	//		 // 2. Loop Body: Use the local clone (localNet) instead of 'this'
	//		 ( i, state, localNet ) => {
	//			 localNet.Forward( data.Source[ i ] );
	//			 localNet.Back( data.Target[ i ] );
	//			 return localNet.Error; // Return local error to be accumulated
	//		 },

	//		 // 3. Local Finally: Safely merge the local error and local gradients
	//		 ( localError ) => {
	//			 lock( lockObject )
	//			 {
	//				 totalError += localError;
	//				 // You would also merge gradients/weights here if needed
	//			 }
	//		 }
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
	void Back( double[] target )
	{
		this.OutputLayer.Back( target );

		HiddenLayers.Reverse();
		HiddenLayers.ForEach( h => h.Back() );
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

			if( testData.Match( index, this.OutputLayer.GetMaxItemIndex() ) ) count++;
		}

		return Tools.Percent( count, testData.Source.Count );
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
		 string message;
	}

	#endregion
}