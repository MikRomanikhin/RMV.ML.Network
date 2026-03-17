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
   /// Current Sum Squared Error in the network.
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
	/// <remarks>
	/// Method performs training for a number of epochs as defined in the settings. At intervals specified by the settings, method evaluates
	/// the model's accuracy on the test set and reports progress. The frequency of evaluation and reporting is determined by the configuration. 	
	/// </remarks>
	/// <param name="trainSet">Training data set</param>
	/// <param name="testSet">Test data set</param>
	public void Train( DataSet trainSet, DataSet testSet )
	{
		for( int epoch = 0; epoch < settings.Iterations; epoch++ )
		{
			double batchErrorSum = 0;

			for( int i = 0; i < settings.Batch; i++ )
			{
				Training( trainSet );

				if( Learning == LearningType.Online ) Update( 1 );

				batchErrorSum += Error;
			}

			if( Learning == LearningType.Batch ) Update( settings.Batch );

			double averageError = batchErrorSum / settings.Batch;

			if( epoch % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				Report( $"Epoch={epoch}, Avg Error={averageError:f3}" );// Time={timer.Elapsed:hh\\:mm\\:ss}" );
				continue;
			}

			var acc = Testing( testSet );
			int total = testSet.Target.Count;

			Report( $"Epoch={epoch}, Avg Error={averageError:f3} Accuracy={Tools.Percent( acc.Count, total ):f2}% \n" );			
		}
	}

	/// <summary>
	/// Trains the Net with a random pattern from the specified dataset.
	/// </summary>
	/// <param name="data">The dataset to train on</param>
	void Training( DataSet data )
	{
		int index = Random.Shared.Next( data.Source.Count ); // random pattern index

		Forward( data.Source[ index ] );

		Back( data.Target[ index ] );		
	}

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
	/// Update weights (non-recursive)
	/// </summary>
	public void Update( int batchSize )
	{	
		foreach( var hidden in this.HiddenLayers ) hidden.Update( batchSize );

		this.OutputLayer.Update( batchSize );
	}

	#endregion


	#region Test ---------------------------------------------------------------

	/// <summary>
	/// Test network against a complete dataset
	/// </summary> 
	public List<int> Testing( DataSet testData )
	{
		var count = new List<int>();

		for( int i = 0; i < testData.Source.Count; i++ )
		{
			int index = MaxItemIndex( testData.Source[ i ] );

			if( index == testData.GetMaxItemIndex( i ) ) count.Add( i );
		}

		return count;
	}

	/// <summary>
	/// Evaluates output for the specified input pattern and returns the index of the output neuron with the highest value.
	/// </summary>
	/// <param name="input">The input pattern</param>
	/// <returns>The index of the output neuron with the highest value</returns>
	int MaxItemIndex( double[] input )
	{
		Forward( input );	

		var nodes = this.OutputLayer.Nodes;		
		double maxValue = nodes[ 0 ].Value;
		int index = 0;

		for( int i = 1; i < nodes.Count; i++ )
		{
			var currentValue = nodes[ i ].Value;

			if( currentValue > maxValue )
			{
				index = i;
				maxValue = currentValue;
			}
		}

		return index;
	}

	#endregion


	#region Report ---------------------------------------------------------------

	public EventHandler<ReportEventArgs> OnReport;
	void TriggerReport( ReportEventArgs ea ) => OnReport?.Invoke( null, ea );

	protected void Report( string message ) => TriggerReport( new ReportEventArgs { Message = message } );
	
	public class  ReportEventArgs : EventArgs
	{
		public string Message { get; set; }
	}

	#endregion
}