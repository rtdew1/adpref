import React from 'react';
import './App.css';
import {
	BrowserRouter as Router,
	Route,
	Switch
} from 'react-router-dom';
import EndTest from "./api/shared/EndTest"
import LoadingPage from "./api/shared/LoadingPage"

export default class App extends React.Component {

	constructor() {
		super();
		this.state = {
			loading: true,
		};
	}

	componentDidMount() {
		fetch(process.env.REACT_APP_BACKEND_ENDPOINT + "/app_info", { method: 'GET' })
		.then(res => res.json())
		// call setState() to make the component render again with new imports
		.then(json => this.setState({
			Home: require(`./api/${json[["paradigm"]]}/Home`).default,
			Initial: require(`./api/${json[["paradigm"]]}/Initial`).default,
			Refine: require(`./api/${json[["paradigm"]]}/Refine`).default,
			Test: require(`./api/${json[["paradigm"]]}/Test`).default,
			loading: false,
		}));
	}

	render() {
		if (this.state.loading) {
			// Render loading state ...
			return <LoadingPage />;
		} else {
			// Render real UI ...
			return (
				<div className="Home">
					<Router>
						<Switch>
							<Route
								exact
								path="/"
								component={this.state.Home}
								render={() => (
									<this.state.Home />
								)}
							/>
							<Route
								exact
								path="/initial"
								component={this.state.Initial}
								render={() => (
									<this.state.Initial />
								)}
							/>
							<Route
								exact
								path="/refine"
								component={this.state.Refine}
								render={() => (
									<this.state.Refine />
								)}
							/>
							<Route
								exact
								path="/test"
								component={this.state.Test}
								render={() => (
									<this.state.Test />
								)}
							/>
							<Route
								exact
								path="/end_test"
								component={EndTest}
								render={() => (
									<EndTest />
								)}
							/>
						</Switch>
					</Router>
				</div>
			);
		}
	}
}

