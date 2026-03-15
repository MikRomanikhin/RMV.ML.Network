using System.Text;

using RMV.ML.Network.Common;

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

   public bool IsInput => this.layerType == LayerType.Input;
   public bool IsHidden => this.layerType == LayerType.Hidden;
   public bool IsOutput => this.layerType == LayerType.Output;

   /// <summary>
   /// Output Layer Error
   /// </summary>
   public double Error { get; private set; }	

	readonly LayerType layerType = LayerType.Unknown;	

	#endregion


	#region Constructor --------------------------------------------------------

	/// <summary>
	/// Constructs a new Layer.
	/// </summary>
	/// <param name="nodes">number of neurons in this layer</param>
	/// <param name="type">layer type</param>
	/// <param name="settings">application settings</param>
	/// <param name="source">previous layer</param>
	/// <param name="activation">activation function</param>
	public Layer( int nodes, LayerType type, AppSettings settings, Layer? source = null, IActivation? activation = null )
	{      
		this.layerType = type;

		this.Nodes = [ .. Enumerable.Range( 0, nodes ).Select( i => new Node( i, settings, activation ) ) ];		

		if( !this.IsInput )
		{
			ArgumentNullException.ThrowIfNull( source );
			this.Source = source;
			source.Target = this;
		}
	}	

	#endregion


	#region Connect ------------------------------------------------------------

	/// <summary>
	/// Recursively connects this to the next Layer
	/// </summary>
	public void Connect()
   {
      if( !this.IsOutput )
      {
         this.Nodes.ForEach( n => n.Connect( this.Target! ) );

         this.Target!.Connect();
      }
   }

   #endregion


   #region Initialize ---------------------------------------------------------    

   /// <summary>
   /// Recoursive initial weights
   /// </summary>
   public void Initialize()
   {
      if( !this.IsInput )  this.Nodes.ForEach( n => n.Initialize() );      

      if( !this.IsOutput ) this.Target!.Initialize();      
   }

	#endregion


	#region FeedForward ------------------------------------------------

	/// <summary>
	/// Feed forward
	/// </summary> 	
	public void Forward( double[] input )
	{
		if( this.IsInput )
		{
			for( int i = 0; i < this.Nodes.Count; i++ ) this.Nodes[ i ].Value = input[ i ];

			this.Target!.Forward( input );
		}
		else if( this.IsHidden )
		{
#if DEBUG
			this.Nodes.ForEach( n => n.Forward() );
#else
	      Parallel.ForEach( this.Nodes, n => n.Forward() );
#endif
			this.Target!.Forward( input );
		}
		else if( this.IsOutput )
		{
#if DEBUG
			this.Nodes.ForEach( n => n.Forward() );
#else
	       Parallel.ForEach( this.Nodes, n => n.Forward() );
#endif
			Softmax();
		}
	}


	/// <summary>
	/// Applies the softmax function to the output layer nodes.
	/// Uses the numerically stable version by subtracting the max value.
	/// </summary>
	void Softmax()
	{
		double max = this.Nodes.Max( n => n.RawValue );

		this.Nodes.ForEach( n => n.Value = Math.Exp( n.RawValue - max ) );

		double sum = this.Nodes.Sum( n => n.Value );

		this.Nodes.ForEach( n => n.Value /= sum );
	}

	#endregion


	#region Back Propagation -------------------------------------------

	/// <summary>
	/// Recursive backwards learning 
	/// </summary>	
	public void Back( double[] output )
	{
		if( this.IsOutput )
		{
			this.Error = this.Nodes.Select( ( n, i ) => n.Back( output[ i ] ) ).Sum();

			this.Source!.Back( output );
		}
		else if( this.IsHidden )
		{
#if DEBUG
	      this.Nodes.ForEach( n => n.Back() );
#else
			Parallel.ForEach( this.Nodes, n => n.Back() );
#endif
			this.Source!.Back( output );
		}
	}

	#endregion


	#region Update Weights -----------------------------------------------------

	/// <summary>
	/// Recoursive update weights 
	/// </summary>
	public void Update( int batchSize )
	{
		if( !this.IsInput )
		{
#if DEBUG
			this.Nodes.ForEach( n => n.Update( batchSize ) );
#else
			Parallel.ForEach( this.Nodes, n => n.Update( batchSize ) );
#endif
		}

		if( !this.IsOutput ) this.Target!.Update( batchSize );
	}
	#endregion


	#region ToString -----------------------------------------------------------

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
