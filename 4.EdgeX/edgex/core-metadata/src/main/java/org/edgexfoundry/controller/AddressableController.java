/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import java.util.List;

import org.edgexfoundry.domain.meta.Addressable;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface AddressableController {

  /**
   * Fetch a specific addressable by database generated id. May return null if no addressable
   * matches on id. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if not found by id.
   * 
   * @param String addressable database generated id
   * @return Addressable
   */
  Addressable addressable(@PathVariable String id);

  /**
   * Return all addressable objects sorted by database generated id. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns LimitExceededException (HTTP 413) if the
   * number returned exceeds the max limit.
   * 
   * @return list of addressable
   */
  List<Addressable> addressables();

  /**
   * Return Addressable with matching name (name should be unique). May be null if none match.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if not found by name.
   * 
   * @param name
   * @return addressable matching on name
   */
  Addressable addressableForName(@PathVariable String name);

  /**
   * Return Addressable objects with given address. List may be empty if none are associated to the
   * address. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param address string (like a URL address)
   * @return list of addressable matching address
   */
  List<Addressable> addressablesByAddress(@PathVariable String address);

  /**
   * Return Addressable objects with given port. List may be empty if none are associated to the
   * port. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param port
   * @return list of addressable matching port
   */
  List<Addressable> addressablesByPort(@PathVariable int port);

  /**
   * Return Addressable objects with given topic. List may be empty if none are associated to the
   * topic. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param topic
   * @return list of addressable matching topic
   */
  List<Addressable> addressablesByTopic(@PathVariable String topic);

  /**
   * Return Addressable objects with given publisher. List may be empty if none are associated to
   * the publisher. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param publisher
   * @return list of addressable matching publisher
   */
  List<Addressable> addressablesByPublisher(@PathVariable String publisher);

  /**
   * Add a new Addressable - name must be unique. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues.
   * 
   * @param Addressable to add
   * @return new database generated id for the new Addressable
   */
  String add(@RequestBody Addressable addressable);

  /**
   * Update the Addressable identified by the id or name in the object provided. Id is used first,
   * name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no addressable with
   * the provided id is found.
   * 
   * @param Addressable object holding the identifier and new values for the Addressable
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody Addressable addressable2);

  /**
   * Remove the Addressable designated by the database generated id for the Addressable. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no addressable with the provided id is found.
   * 
   * @param database generated id
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the Addressable designated by unique name identifier. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * addressable with the provided name is found.
   * 
   * @param unique name of the Addressable
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);

}
