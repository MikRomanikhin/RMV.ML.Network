using System.Text;

using RMV.ML.Network.Configaration;

namespace RMV.ML.Network.Domain;

/// <summary>
/// A layer as a collection of neurons. An input Layer, an output layer and one or more hidden layers.
/// </summary>  
public class Layer
{

   #region Properties ---------------------------------------------------------  

	/// <summary>
	/// Neurons present in this Layer.
	/// </summary>
	public List<Node> Nodes { get; private set; }

   /// <summary>
   /// The Layer feeding this Layer.
   /// </summary>
   public Layer? Source { get; private set; }

   /// <summary>
   /// The Layer fed by this Layer.
   /// </summary>
   public Layer? Target { get; private set; }  

   /// <summary>
   /// Output Layer Error
   /// </summary>
   public double Error { get; private set; }

	/// <summary>
	/// Helpers to identify the Layer type
	/// </summary>
	bool IsInput => this.layerType == LayerType.Input;
	bool IsHidden => this.layerType == LayerType.Hidden;
	bool IsOutput => this.layerType == LayerType.Output;

	#endregion


	#region Constructor --------------------------------------------------------

	readonly LayerType layerType = LayerType.Unknown;
	readonly IActivation? activation;

	/// <summary>
	/// Constructs a new Layer.
	/// </summary>
	/// <param name="nodes">number of neurons in this layer</param>
	/// <param name="type">layer type</param>
	/// <param name="settings">application settings</param>
	/// <param name="source">previous layer</param>	
	public Layer( int nodes, LayerType type, AppSettings settings, Layer? source = null )
	{
		this.layerType = type;

		if( this.IsHidden ) this.activation = new Relu();
		if( this.IsOutput ) this.activation = new Identity();		

		this.Nodes = [ .. Enumerable.Range( 0, nodes ).Select( i => new Node( i, settings, activation ) ) ];

		if( !this.IsInput )
		{
			ArgumentNullException.ThrowIfNull( source );
			this.Source = source;
			source.Target = this;
		}
	}

	public Layer()
	{
		this.Nodes = [];
	}

	//public Layer Clone()
	//{
	//	var clone = new Layer ;
		
	//	for( int i = 0; i < this.Nodes.Count; i++ )
	//	{
	//		var node = this.Nodes[ i ];
	//		var cloneNode = clone.Nodes[ i ];
	//		cloneNode.Value = node.Value;
	//		cloneNode.RawValue = node.RawValue;
	//		cloneNode.Error = node.Error;
	//		for( int j = 0; j < node.In.Count; j++ )
	//		{
	//			var edge = node.In[ j ];
	//			var cloneEdge = cloneNode.In[ j ];
	//			cloneEdge.Weight = edge.Weight;
	//			cloneEdge.DeltaWeight = edge.DeltaWeight;
	//		}
	//	}
	//	return clone;
	//}

	#endregion


	#region Connect/Initialize -------------------------------------------------

	/// <summary>
	/// Iteratively connects this layer to the next one. 
	/// </summary>
	public void Connect() => this.Nodes.ForEach( n => n.Connect( this.Target! ) );	
		  	

	/// <summary>
	/// Iteratively initialize weights
	/// </summary>
	public void Initialize() => this.Nodes.ForEach( n => n.HeInitialize() );

	#endregion


	#region Feed Forward -------------------------------------------------------	

	/// <summary>	
	/// Input layer feed forward  with the specified input pattern.
	/// </summary>
	/// <param name="input">The input pattern</param>
	public void Forward( double[] input )
	{
//#if DEBUG
		for( int i = 0; i < this.Nodes.Count; i++ ) this.Nodes[ i ].Value = input[ i ];
//#else
//		Parallel.For( 0, this.Nodes.Count, i => this.Nodes[ i ].Value = input[ i ] );
//#endif
	}

	/// <summary>
	/// Hidden and Output layer feed forward
	/// </summary>
	public void Forward() => this.Nodes.ForEach( n => n.Forward() );
//	{
//#if DEBUG
//		this.Nodes.ForEach( n => n.Forward() );
//#else
//		Parallel.ForEach( this.Nodes, n => n.Forward() );
//#endif
//		if( this.IsOutput ) Softmax();
//	}

	/// <summary>
	/// Applies the softmax function to the output layer nodes	
	/// </summary>
	void Softmax()
	{		
		double max = this.Nodes.Max( n => n.RawValue );

		this.Nodes.ForEach( n => n.Value = Math.Exp( n.RawValue - max ) );

		double sum = this.Nodes.Sum( n => n.Value );		

		this.Nodes.ForEach( n => n.Value /= sum );
	}

#endregion


	#region Back Propagation ---------------------------------------------------	

	/// <summary>
	/// Output layer learning with the specified output pattern and error calculation.
	/// </summary>
	/// <param name="output">The output pattern</param>
	public void Back( double[] output )
	{
		this.Error = this.Nodes.Select( ( n, i ) => n.Backward( output[ i ] ) ).Sum();
	}

	/// <summary>
	/// Hidden layer learning with error calculation based on the target layer errors and weights.
	/// </summary>
	public void Back() => this.Nodes.ForEach( n => n.Backward() );
//	{
//#if DEBUG
//		this.Nodes.ForEach( n => n.Back() );
//#else
//		Parallel.ForEach( this.Nodes, n => n.Back() );
//#endif
//	}

	#endregion


	#region Update Weights -----------------------------------------------------	

	/// <summary>
	/// Update weights 
	/// </summary>
	public void Update( int batchSize )
	{	
#if DEBUG
		this.Nodes.ForEach( n => n.Update( batchSize ) );
#else
		Parallel.ForEach( this.Nodes, n => n.Update( batchSize ) );
#endif
	}

	#endregion


	#region Misc ---------------------------------------------------------------

	/// <summary>
	/// Calculates the index of the output node with the highest value
	/// </summary>	
	public int GetMaxItemIndex()
	{
		int maxIndex = 0;
		double maxValue = this.Nodes[ 0 ].Value;

		for( int i = 1; i < this.Nodes.Count; i++ )
		{
			double value = this.Nodes[ i ].Value;

			if( value > maxValue )
			{
				maxValue = value;
				maxIndex = i;
			}
		}

		return maxIndex;
	}


	public override string ToString()
   {
      int count = 0;
      var sb = new StringBuilder();

      sb.AppendLine().Append( "Layer " ).Append( count ).Append( " Type: " ).Append( this.layerType );

      if( this.IsHidden || this.IsOutput )
      {
         sb.Append( " Source: " ).Append( this.Source!.layerType );
			sb.Append( " Target: " ).Append( this.Target!.layerType );
		}     

      sb.AppendLine();

      this.Nodes.ForEach( n => sb.AppendLine( n.ToString() ) );

      return sb.ToString();
   }

	#endregion

} 
