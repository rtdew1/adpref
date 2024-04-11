import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Conducts the refining ratings - button submits rating and loads next
class Refine extends React.Component {

  static propTypes = {
    cookies: instanceOf(Cookies).isRequired
  };

  constructor(props) {
    super(props);
    const { cookies } = props;
    this.state = {
      img: "",
      user: cookies.get('user'),
      img_fallback: "https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.1/css/loader.gif",
      arbitrary_check: -1,
      arbitrary_count: 0
    }
    this.handleSubmit = this.handleSubmit.bind(this);
    this.arbitraryImageURLs = [
      'https://s7d2.scene7.com/is/image/nu/4130059680055_049_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130024090046_095_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130077370111_049_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/36521532_012_b?$pdp-regular$'
    ]
  }

  // React function that is called when the page loads.
  componentDidMount() {
    // Send an HTTP request to the server.
    fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + '/refine_send/' + this.state.user, { method: 'GET' } // The type of HTTP request.
    ).then(
      res => {
        // Convert the response data to a JSON.
        return res.json();
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
    ).then
      (data => {
        // Act on data
        if (data.move_on === true) {
          // Move on to testing stage
          this.nextStage()
        }
        else {
          this.setState({
            // Load next image
            img: data.img_url,
            arbitrary_check: data.arbitrary_check,
            arbitrary_count: this.state.arbitrary_count + (data.arbitrary_check !== -1 ? 1 : 0)
          });

          if (this.state.arbitrary_check !== -1) {
            this.setState({
              img: this.arbitraryImageURLs[this.state.arbitrary_count % this.arbitraryImageURLs.length]
            })
          }
        }
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
      );
  }

  timeout(delay) {
    return new Promise(res => setTimeout(res, delay));
  }

  // Finish the refining stage
  async nextStage() {
    // Call ahead to calculate testing items
    try {
      await fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/calc_results/' + this.state.user, {
        method: 'POST',
        body: "",
      })

      const res = (await fetch(
        process.env.REACT_APP_BACKEND_ENDPOINT + '/num_testing/' + this.state.user, { method: 'GET' }
      )).json()

      this.setState({
        num_testing: res.num_testing
      })

      window.location.href = "/test"
    } catch (error) {
      console.log(error)
    }
  }

  // Submit the rating to model and refresh
  async handleSubmit(event) {
    event.preventDefault();
    const data = new FormData(event.target);

    // Unload current image and rating
    document.getElementById('rating').value = "";
    this.setState({
      img: ""
    });

    // Send rating to model and await calculation
    fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/refine_receive/' + this.state.user, {
      method: 'POST',
      body: data,
    }).then(
      res => { this.componentDidMount(); },
      err => { console.log(err) }
    );
  }

  render() {
    let page_body;
    if (this.state.img === "") {
      page_body = (
        <div>
          <h2>Loading</h2>
          <img src={this.state.img_fallback} className="Loading" alt="Loading next dress" />
        </div>
      )
    } else {
      page_body = (
        <div>
          <header className="App-header">
            <p>Please rate the following dress
              {
                this.state.arbitrary_check === -1 ?
                  " on a scale from 0 to 10 (higher is better):" :
                  " a " + this.state.arbitrary_check + ":"
              }
            </p>
            <img src={this.state.img} className="App-photo" alt="Please rate this dress." />
            <br />
            <form onSubmit={this.handleSubmit}>
              <label htmlFor="rating">Enter your rating for this dress</label>
              <br />
              <input id="rating" name="rating" type="number" step="0.1" min="0" max="10" required />
              <button className="App-button">Submit</button>
              <br />
            </form>
          </header>
        </div>)
    }
    return (
      <div className="App">
        <header className="App-header">
          {page_body}
        </header>
      </div>
    );
  }
}

export default withCookies(Refine);