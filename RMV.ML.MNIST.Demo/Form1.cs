using RMV.ML.Network.Common;
using RMV.ML.MNIST.CS;

namespace RMV.ML.MNIST.Demo;

/// <summary>
/// MNIST Visualization
/// </summary>
public partial class View : Form
{
	readonly List<int[]> images; //input data storage
	readonly List<int> actual;   //actual labels for images
	readonly List<int> errors; //
	readonly List<int> indexes; //

	public View()
	{
		InitializeComponent();

		var settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );

		(this.images, this.actual) = Parser.GetImages( File.ReadAllLines( settings.TestPath ) );
		errors = [ .. File.ReadAllText( settings.ErrorPath ).Split( ',' ).Select( int.Parse ) ];
		indexes = [ .. File.ReadAllText( settings.IndexPath ).Split( ',' ).Select( int.Parse ) ];

		int first = this.indexes[ 0 ];

		this.pictureBox.Image = GetBitmap( first );
		this.textActual.Text = this.actual[ first ].ToString();
		this.textPredict.Text = this.errors[ 0 ].ToString();
	}


	/// <summary>
	/// Displays current image
	/// </summary>
	void SlectImage_Click( object sender, EventArgs e )
	{
		int ind = int.Parse( SelectImage.Text ); // index from text box
		int index = this.indexes[ ind ];         // image index from indexes list		

		this.pictureBox.Image = GetBitmap( index );
		this.textActual.Text = this.actual[ index ].ToString();
		this.textPredict.Text = this.errors[ ind ].ToString();
	}


	/// <summary>
	/// Converts data to Bitmap
	/// </summary>
	/// <param name="data">data</param>
	/// <returns>Bitmap</returns>
	Bitmap GetBitmap( int index = 0 )
	{
		var data = this.images[ index ].Reshape( 28, 28 );

		var bitmap = new Bitmap( this.pictureBox.Width, this.pictureBox.Height );

		var brush = new SolidBrush( Color.White );

		using( var g = Graphics.FromImage( bitmap ) )
		{
			g.FillRectangle( brush, new Rectangle( new Point( 0, 0 ), new Size( this.pictureBox.Width, this.pictureBox.Height ) ) );

			int stepX = this.pictureBox.Width / 28;
			int stepY = this.pictureBox.Height / 28;

			var size = new Size( stepX, stepY );

			for( int i = 0; i < 27; i++ )
			{
				for( int j = 0; j < 27; j++ )
				{
					int pixelColor = 255 - data[ j, i ];

					brush.Color = Color.FromArgb( pixelColor, pixelColor, pixelColor );

					var point = new Point( stepX * i, stepY * ( 1 + j ) );

					g.FillRectangle( brush, new Rectangle( point, size ) );
				}
			}
		}

		return bitmap;
	}

	
}
