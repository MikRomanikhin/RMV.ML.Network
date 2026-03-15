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
		this.InputLayer = new Layer( settings.Input, LayerType.Input, settings );      
      var lastLayer = InputLayer;

      for( int i = 0; i < settings.Hidden.Length; i++ )
		{
			var hiddenLayer = new Layer( settings.Hidden[i], LayerType.Hidden, settings, lastLayer, new Relu() );
			this.HiddenLayers.Add( hiddenLayer );
			lastLayer = hiddenLayer;			
		}

		this.OutputLayer = new Layer( settings.Output, LayerType.Output, settings, lastLayer, new Identity() );			
	}	   

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
   public void Connect() => this.InputLayer.Connect();   

   /// <summary>
   /// Initialisation
   /// </summary>
   public void Initialize() => this.InputLayer.Initialize();

	#endregion


	#region Training -----------------------------------------------------------

	/// <summary>
	/// Trains the Net with a random pattern from the specified dataset.
	/// </summary>
	/// <param name="data">The dataset to train on</param>
	public void Training( DataSet data )
	{
		int index = Random.Shared.Next( data.Source.Count ); // random pattern index
		
		this.InputLayer.Forward( data.Source[ index ] );

		this.OutputLayer.Back( data.Target[ index ] );
	}	

	/// <summary>
	/// Update weights 
	/// </summary>
	public void Update( int batchSize ) => this.InputLayer.Update( batchSize );

	#endregion


	#region Test ---------------------------------------------------------------

	/// <summary>
	/// Test network against a complete dataset
	/// </summary> 
	public List<int> Testing( DataSet testData )
	{
		var errors = new List<int>();

		for( int i = 0; i < testData.Source.Count; i++ )
		{
			int index = MaxItemIndex( testData.Source[ i ] );

			if( index != testData.GetMaxItemIndex( i ) ) errors.Add( i );
		}

		return errors;
	}

	/// <summary>
	/// Evaluates output for the specified input pattern and returns the index of the output neuron with the highest value.
	/// </summary>
	/// <param name="input">The input pattern</param>
	/// <returns>The index of the output neuron with the highest value</returns>
	int MaxItemIndex( double[] input )
	{
		this.InputLayer.Forward( input );

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

}