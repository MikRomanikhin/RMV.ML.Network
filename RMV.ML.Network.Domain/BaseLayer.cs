
namespace RMV.ML.Network.Domain;

abstract class BaseLayer
{
	#region Properties ---------------------------------------------------------  

	/// <summary>
	/// Neurons present in this Layer.
	/// </summary>
	public List<Node> Nodes { get; private set; }

	/// <summary>
	/// The Layer feeding this Layer.
	/// </summary>
	public BaseLayer? Source { get; private set; }

	/// <summary>
	/// The Layer fed by this Layer.
	/// </summary>
	public BaseLayer? Target { get; private set; }	

	#endregion


	#region Constructor --------------------------------------------------------

	
	protected IActivation? activation;

	/// <summary>
	/// Constructs a new Layer.
	/// </summary>
	/// <param name="nodes">number of neurons in this layer</param>
	/// <param name="type">layer type</param>
	/// <param name="settings">application settings</param>
	/// <param name="source">previous layer</param>	
	public BaseLayer( int nodes, AppSettings settings, BaseLayer? source = null )
	{		
		this.Nodes = [ .. Enumerable.Range( 0, nodes ).Select( i => new Node( i, settings, activation ) ) ];

		if( source != null )
		{
			this.Source = source;
			source.Target = this;
		}		
	}


	#endregion


	#region Connect/Initialize -------------------------------------------------

	/// <summary>
	/// Iteratively connects this layer to the next one. 
	/// </summary>
	internal void Connect() => this.Nodes.ForEach( n => n.Connect( this.Target! ) );


	/// <summary>
	/// Iteratively initialize weights
	/// </summary>
	internal void Initialize() => this.Nodes.ForEach( n => n.BmInitialize() );

	#endregion


	#region Forward / Backward -------------------------------------------------
	
	public abstract void Forward( double[]? input = null );

	public abstract void Backward( double[]? output = null );	

	#endregion	


	#region Update Weights -----------------------------------------------------	

	/// <summary>
	/// Update weights 
	/// </summary>
	internal void Update( int batchSize )
	{
#if DEBUG
		this.Nodes.ForEach( n => n.Update( batchSize ) );
#else
		Parallel.ForEach( this.Nodes, n => n.Update( batchSize ) );
#endif
	}

	#endregion
	
}
