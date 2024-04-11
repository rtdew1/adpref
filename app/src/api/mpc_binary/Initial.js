import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Conducts the initial ratings - button submits rating and loads next
class Initial extends React.Component {

  static propTypes = {
    cookies: instanceOf(Cookies).isRequired
  };

  constructor(props) {
    super(props);
    const { cookies } = props;
    this.state = {
      img_1: "",
      img_2: "",
      user: cookies.get('user'),
      img_fallback: "https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.1/css/loader.gif"
    }
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  // React function that is called when the page loads.
  async componentDidMount() {
    // Send an HTTP request to the server.

    fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + '/initial_send/' + this.state.user, { method: 'GET' } // The type of HTTP request.
    ).then(
      res => {
        // Convert the response data to a JSON.
        return res.json();
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
    ).then(
      data => {
        // Act on data
        if (data.move_on === true) {
          // Move on to refining stage
          this.nextStage();
        }
        else {
          this.setState({
            // Load next image
            img_1: data.img_url_1,
            img_2: data.img_url_2
          });
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

  // Finish the initial stage
  async nextStage() {
    // Call ahead to build the model

    try {
      await fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/build_model/' + this.state.user, {
        method: 'POST',
        body: "",
      })
      window.location.href = "/refine"
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
      img_1: "",
      img_2: ""
    });

    // Send rating to model and await calculation
    fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/initial_receive/' + this.state.user, {
      method: 'POST',
      body: data,
    }).then(
      res => { this.componentDidMount(); },
      err => { console.log(err) }
    );
  }

  render() {
    let page_body;

    if (this.state.img_1 === "" || this.state.img_2 === "") {
      page_body = (
        <div>
          <h2>Loading</h2>
          <img src={this.state.img_fallback} className="Loading" alt="Loading next dress" />
        </div>
      )
    } else {
      page_body = (
        <div>
          <p>Which of these two dresses do you prefer?</p>
          <div className="img-container">
            <img src={this.state.img_1} className="img-left" alt="Please rate this dress." />
            &nbsp; or &nbsp;
            <img src={this.state.img_2} className="img-right" alt="Please rate this dress." />
          </div>
          <br />
          <form onSubmit={this.handleSubmit}>
            <label htmlFor="rating">Enter your preference for the dresses</label>
            <br />
            <table className="likert"> 
                  <thead>
                    <tr>
                      {/*<th> Strongly Prefer Left </th>*/}
                      <th> Prefer left </th>
                      {/*<th> Indifferent  </th>*/}
                      <th> Prefer Right </th>
                      {/*<th> Strongly Prefer Right </th>*/}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      {/*<td> <input type="radio" id="rating" name="rating" value="-1" required /> </td>*/}
                      <td> <input type="radio" id="rating" name="rating" value="-1" required /> </td>
                      {/*<td> <input type="radio" id="rating" name="rating" value="0" required /> </td>*/}
                      <td> <input type="radio" id="rating" name="rating" value="1" required /> </td>
                      {/*<td> <input type="radio" id="rating" name="rating" value="1" required /> </td>*/}
                    </tr>
                  </tbody>
                </table>
            <button className="mpc-submit-button">Submit</button>
            <br />
          </form>
        </div>
      )
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

export default withCookies(Initial);
