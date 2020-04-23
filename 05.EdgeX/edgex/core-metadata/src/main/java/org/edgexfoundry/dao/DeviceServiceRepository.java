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

package org.edgexfoundry.dao;

import java.util.List;

import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.DeviceService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface DeviceServiceRepository extends MongoRepository<DeviceService, String> {

  DeviceService findByName(String name);

  List<DeviceService> findByAddressable(Addressable addressable);

  Page<DeviceService> findByAddressable(Addressable addressable, Pageable pageable);

  List<DeviceService> findByLabelsIn(String label);

  Page<DeviceService> findByLabels(String label, Pageable pageable);

}
